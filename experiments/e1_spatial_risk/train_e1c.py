#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SpatialRiskWindowDataset
from model import SpatialRiskConvLSTM

def parse_args():
    p=argparse.ArgumentParser("Train E1-C localization-aware ConvLSTM")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="+", required=True)
    p.add_argument("--history", type=int, default=15)
    p.add_argument("--horizons", nargs="+", type=int, default=[3,5,10,15])
    p.add_argument("--window-stride", type=int, default=1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument("--loss-mode", choices=["weighted_l1","l1_dice","l1_bce_dice"], default="l1_bce_dice")
    p.add_argument("--lambda-l1", type=float, default=1.0)
    p.add_argument("--lambda-high-risk", type=float, default=2.0)
    p.add_argument("--lambda-bce", type=float, default=0.5)
    p.add_argument("--lambda-dice", type=float, default=0.5)
    p.add_argument("--high-risk-weight", type=float, default=4.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["float16","bfloat16"], default="float16")
    p.add_argument("--early-stop-patience", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("results/V6/e1_spatial_risk/e1C_convlstm_localization.pt"))
    p.add_argument("--verbose", type=int, default=2)
    return p.parse_args()

def choose_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class LocalizationAwareLoss(nn.Module):
    def __init__(self, mode="l1_bce_dice", risk_threshold=0.5,
                 lambda_l1=1.0, lambda_high_risk=2.0, lambda_bce=0.5,
                 lambda_dice=0.5, high_risk_weight=4.0, eps=1e-6):
        super().__init__()
        self.mode=mode; self.risk_threshold=float(risk_threshold)
        self.lambda_l1=float(lambda_l1); self.lambda_high_risk=float(lambda_high_risk)
        self.lambda_bce=float(lambda_bce); self.lambda_dice=float(lambda_dice)
        self.high_risk_weight=float(high_risk_weight); self.eps=float(eps)

    def forward(self, pred, target):
        abs_err=(pred-target).abs()
        high_mask=(target>=self.risk_threshold).float()
        weights=1.0 + high_mask*(self.high_risk_weight-1.0)
        weighted_l1=(abs_err*weights).mean()
        base_l1=abs_err.mean()
        high_risk_l1=(abs_err*high_mask).sum()/(high_mask.sum()+self.eps)

        zero=torch.zeros((), device=pred.device)
        if self.mode=="weighted_l1":
            total=self.lambda_l1*weighted_l1
            return total, {"total":total.detach(),"l1":base_l1.detach(),
                           "weighted_l1":weighted_l1.detach(),"high_risk_l1":high_risk_l1.detach(),
                           "bce":zero,"dice_loss":zero}

        temperature=0.10
        logits=(pred-self.risk_threshold)/temperature
        pred_prob=torch.sigmoid(logits)
        target_binary=high_mask

        # AMP-safe BCE: operate on logits rather than sigmoid probabilities.
        bce=nn.functional.binary_cross_entropy_with_logits(
            logits,
            target_binary,
        )

        dims=tuple(range(2,pred_prob.ndim))
        inter=(pred_prob*target_binary).sum(dim=dims)
        denom=pred_prob.sum(dim=dims)+target_binary.sum(dim=dims)
        dice=(2.0*inter+self.eps)/(denom+self.eps)
        has_positive=target_binary.sum(dim=dims)>0
        dice_loss=(1.0-dice[has_positive].mean()) if has_positive.any() else zero

        if self.mode=="l1_dice":
            total=self.lambda_l1*weighted_l1 + self.lambda_high_risk*high_risk_l1 + self.lambda_dice*dice_loss
        else:
            total=self.lambda_l1*weighted_l1 + self.lambda_high_risk*high_risk_l1 + self.lambda_bce*bce + self.lambda_dice*dice_loss
        return total, {"total":total.detach(),"l1":base_l1.detach(),
                       "weighted_l1":weighted_l1.detach(),"high_risk_l1":high_risk_l1.detach(),
                       "bce":bce.detach(),"dice_loss":dice_loss.detach()}

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              use_amp=False, amp_dtype=torch.float16, verbose=2, stage="train"):
    training=optimizer is not None
    model.train(training)
    sums={}; n=0
    it=tqdm(loader, desc=stage, leave=False, disable=(verbose<2))
    for batch in it:
        x=batch["x"].to(device, non_blocking=True)
        y=batch["y"].to(device, non_blocking=True)
        bs=x.size(0)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred=model(x); loss,parts=criterion(pred,y)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    optimizer.step()
        for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v.item())*bs
        n+=bs
        if verbose>=2:
            it.set_postfix(loss=f"{loss.item():.4f}",l1=f"{parts['l1'].item():.4f}",
                           hi=f"{parts['high_risk_l1'].item():.4f}",dice=f"{parts['dice_loss'].item():.4f}")
    return {k:v/max(n,1) for k,v in sums.items()}

def main():
    args=parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    device=choose_device(); print(f"[device] {device}")
    train_ds=SpatialRiskWindowDataset(args.train,args.history,tuple(args.horizons),args.window_stride)
    val_ds=SpatialRiskWindowDataset(args.val,args.history,tuple(args.horizons),args.window_stride)
    print(f"[data] train windows={len(train_ds):,}")
    print(f"[data] val windows  ={len(val_ds):,}")

    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
                            pin_memory=device.type=="cuda",persistent_workers=args.num_workers>0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                          pin_memory=device.type=="cuda",persistent_workers=args.num_workers>0)

    model=SpatialRiskConvLSTM(args.hidden,len(args.horizons)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=2,min_lr=1e-5)
    criterion=LocalizationAwareLoss(args.loss_mode,args.risk_threshold,args.lambda_l1,args.lambda_high_risk,
                                    args.lambda_bce,args.lambda_dice,args.high_risk_weight)

    use_amp=bool(args.amp and device.type=="cuda")
    amp_dtype=torch.float16 if args.amp_dtype=="float16" else torch.bfloat16
    scaler_enabled=use_amp and amp_dtype==torch.float16
    try:
        scaler=torch.amp.GradScaler("cuda",enabled=scaler_enabled)
    except TypeError:
        scaler=torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    print(f"[loss] mode={args.loss_mode}")
    print(f"[loss] threshold={args.risk_threshold} high_risk_weight={args.high_risk_weight}")
    print(f"[loss] lambdas: l1={args.lambda_l1} high={args.lambda_high_risk} bce={args.lambda_bce} dice={args.lambda_dice}")
    print(f"[amp] enabled={use_amp} dtype={args.amp_dtype}")

    history=[]; best=float("inf"); bad=0
    for epoch in range(1,args.epochs+1):
        t0=time.time()
        tr=run_epoch(model,train_loader,criterion,device,optimizer,scaler,use_amp,amp_dtype,args.verbose,f"train {epoch:02d}")
        va=run_epoch(model,val_loader,criterion,device,None,None,use_amp,amp_dtype,args.verbose,f"val {epoch:02d}")
        scheduler.step(va["total"]); elapsed=time.time()-t0; lr=optimizer.param_groups[0]["lr"]
        history.append({"epoch":epoch,"lr":lr,"epoch_seconds":elapsed,"train":tr,"val":va})
        print(f"[epoch {epoch:03d}] train={tr['total']:.6f} val={va['total']:.6f} "
              f"val_l1={va['l1']:.6f} val_highrisk={va['high_risk_l1']:.6f} "
              f"val_dice={va['dice_loss']:.6f} lr={lr:.2e} time={elapsed:.1f}s")
        if va["total"]<best:
            best=va["total"]; bad=0
            torch.save({"model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "epoch":epoch,"best_val_loss":best,
                        "config":vars(args),
                        "loss_config":{"mode":args.loss_mode,"risk_threshold":args.risk_threshold,
                                       "lambda_l1":args.lambda_l1,"lambda_high_risk":args.lambda_high_risk,
                                       "lambda_bce":args.lambda_bce,"lambda_dice":args.lambda_dice,
                                       "high_risk_weight":args.high_risk_weight}}, args.output)
            print(f"[save] best checkpoint -> {args.output}")
        else:
            bad+=1
        args.output.with_suffix(".history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        if args.early_stop_patience>0 and bad>=args.early_stop_patience:
            print(f"[early-stop] no validation improvement for {bad} epochs")
            break
    print(f"[done] best validation loss={best:.6f}")
    print(f"[done] checkpoint={args.output}")

if __name__=="__main__":
    main()
