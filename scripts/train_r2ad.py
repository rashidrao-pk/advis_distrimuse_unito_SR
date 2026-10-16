#!/usr/bin/env python3
import argparse,sys,json,random,csv
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.core import build_models
from r2ad.dataset import RectangleDataset
from r2ad.mcap_data import McapRectangleDataset
from r2ad.config import resolve, DEFAULT_CONFIG
from tqdm.auto import tqdm


def save_probe_grid(probe, enc, dec, device, path):
    """Fixed RGB inputs + deterministic reconstructions + absolute error maps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    was_enc, was_dec = enc.training, dec.training
    enc.eval(); dec.eval()
    with torch.inference_mode():
        x = probe.to(device, non_blocking=True)
        mu, _ = enc(x)
        y = dec(mu)
        errs = (x - y).abs().mean(dim=1)
        x_np = ((x.detach().cpu().float().permute(0,2,3,1).numpy()+1)/2).clip(0,1)
        y_np = ((y.detach().cpu().float().permute(0,2,3,1).numpy()+1)/2).clip(0,1)
        e_np = (errs.detach().cpu().float().numpy()/2).clip(0,1)
    rows = len(x_np)
    fig, axes = plt.subplots(rows, 3, figsize=(9, max(2.7*rows,3)), squeeze=False)
    for i in range(rows):
        axes[i,0].imshow(x_np[i]); axes[i,1].imshow(y_np[i]); axes[i,2].imshow(e_np[i],cmap='inferno',vmin=0,vmax=0.5)
        for ax in axes[i]:ax.axis('off')
        axes[i,0].set_ylabel(f'Probe {i+1}', fontsize=9)
    for j,t in enumerate(('Fixed original patch','Reconstructed (mu)','Absolute error (0..0.5)')):
        axes[0,j].set_title(t,fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=125)
    plt.close(fig)
    enc.train(was_enc); dec.train(was_dec)


def save_loss_plot(records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for key in ('reconstruction','kl','generator_adversarial','discriminator'):
        ax.plot([r['step'] for r in records],[r[key] for r in records],label=key,alpha=.8)
    ax.set(xlabel='Global optimizer steps',ylabel='Loss',title='R2AD training diagnostics')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)


def save_metrics(records, path):
    with path.open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=('epoch','step','batch','reconstruction','kl','generator_adversarial','discriminator','generator_total'))
        w.writeheader();w.writerows(records)

def main():
    p=argparse.ArgumentParser(description='Train R2AD VAE-GAN using normal full-resolution images')
    p.add_argument('--dataset-source',choices=('images','mcap'),default='images')
    p.add_argument('--manifest',help='MCAP manifest from index_rosbags_r2ad.py')
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--safety-area',required=True,choices=('PLeft','PRight','RoboArm','ConvBelt','SafetyZone'))
    for flag in ('train-dir','val-dir','mask','output'):p.add_argument('--'+flag)
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--patch-size',type=int,default=224);p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--samples-per-frame',type=int,default=6);p.add_argument('--latent-dims',type=int)
    p.add_argument('--lr',type=float,default=.0002);p.add_argument('--workers',type=int)
    p.add_argument('--kl-weight',type=float,default=.0001);p.add_argument('--adv-weight',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    p.add_argument('--monitor-every',type=int,default=100,help='Save monitoring images every N optimizer steps; 0 disables')
    p.add_argument('--monitor-probes',type=int,default=4,help='Number of fixed validation patches to display')
    p.add_argument('--monitor-dir',default=None,help='Defaults to checkpoint parent/live_monitor')
    args=p.parse_args()
    if args.dataset_source=='mcap':
        from r2ad.config import load_config
        cfg=load_config(args.config);root=Path(cfg['data']['dataset_base']).expanduser();rcfg=cfg.get('r2ad') or {}
        mask_cfg=(rcfg.get('safety_zone_mask') or cfg.get('data',{}).get('mask_types',{}).get('SafetyZone')) if args.safety_area=='SafetyZone' else cfg['data']['mask_types'].get(args.safety_area)
        args.mask=args.mask or mask_cfg
        if not args.mask:raise ValueError('Missing safety mask in config')
        args.mask=str((root/args.mask) if not Path(args.mask).is_absolute() else args.mask)
        args.manifest=args.manifest or rcfg.get('manifest') or str(root/'r2ad'/'manifest.csv')
        args.manifest=str((root/args.manifest) if not Path(args.manifest).is_absolute() else args.manifest)
        base=Path(cfg.get('results',{}).get('output_dir','results'))/str(rcfg.get('dataset_version','V6'))/'r2ad'/args.safety_area
        args.output=args.output or str(base/'model_r2ad.pt')
        args.latent_dims=args.latent_dims or int(cfg.get('models',{}).get('latent_dims',64))
        args.epochs=args.epochs or int(cfg.get('training',{}).get('epochs',50))
        args.batch_size=args.batch_size or int(cfg.get('training',{}).get('batch_size',32))
        args.workers=args.workers if args.workers is not None else int(cfg.get('training',{}).get('num_workers',4))
        if not Path(args.manifest).is_file():raise FileNotFoundError(args.manifest)
        if not Path(args.mask).is_file():raise FileNotFoundError(args.mask)
    else:args=resolve(args,'train')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.dataset_source=='mcap':
        train=McapRectangleDataset(args.manifest,'train',args.mask,args.patch_size,args.min_coverage,args.samples_per_frame,True,args.seed)
        val=McapRectangleDataset(args.manifest,'val',args.mask,args.patch_size,args.min_coverage,1,False,args.seed)
    else:
        train=RectangleDataset(args.train_dir,args.mask,args.patch_size,args.min_coverage,args.samples_per_frame,True,args.seed)
        val=RectangleDataset(args.val_dir,args.mask,args.patch_size,args.min_coverage,1,False,args.seed)
    tr=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device=='cuda')
    va=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device=='cuda')
    enc,dec,disc=build_models(args.latent_dims,device)
    opt_g=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=args.lr)
    opt_d=torch.optim.Adam(disc.parameters(),lr=args.lr/2)
    bce=nn.BCEWithLogitsLoss();best=float('inf');out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if args.monitor_every < 0 or args.monitor_probes < 1:
        p.error('--monitor-every must be >= 0 and --monitor-probes must be >= 1')
    mon_dir=Path(args.monitor_dir) if args.monitor_dir else out.parent/'live_monitor'
    history=[]; global_step=0; probe=None
    if args.monitor_every:
        mon_dir.mkdir(parents=True,exist_ok=True)
        # The same patches are held in memory for the entire run.
        probe_count=min(args.monitor_probes,len(val))
        probe=torch.stack([val[i] for i in range(probe_count)],dim=0)
        save_probe_grid(probe,enc,dec,device,mon_dir/'probe_step_000000.png')
        print(f'[monitor] fixed probes={probe_count} | output={mon_dir}',flush=True)
    print(f'[data] train frames={len(train.rows) if args.dataset_source=="mcap" else len(train.paths)}, val frames={len(val.rows) if args.dataset_source=="mcap" else len(val.paths)}, train patches/epoch={len(train)}')
    for epoch in tqdm(range(1,args.epochs+1),desc='Epochs'):
        enc.train();dec.train();disc.train();running=[]
        for batch_idx,x in enumerate(tqdm(tr,desc=f'Epoch {epoch} batches',leave=False),start=1):
            x=x.to(device,non_blocking=True)
            mu,logvar=enc(x)
            z=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
            recon=dec(z)
            opt_d.zero_grad(set_to_none=True)
            logits_real=disc(x)
            logits_fake=disc(recon.detach())
            dloss=.5*(bce(logits_real,torch.ones_like(logits_real))+bce(logits_fake,torch.zeros_like(logits_fake)))
            dloss.backward();opt_d.step()
            for par in disc.parameters():par.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            rloss=F.mse_loss(recon,x)
            kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(dim=1).mean()/args.latent_dims
            pred_fake=disc(recon)
            adv=bce(pred_fake,torch.ones_like(pred_fake))
            gloss=rloss+args.kl_weight*kl+args.adv_weight*adv
            gloss.backward();opt_g.step()
            for par in disc.parameters():par.requires_grad_(True)
            running.append(float(rloss.item()))
            global_step+=1
            if args.monitor_every:
                rec=dict(epoch=epoch,step=global_step,batch=batch_idx,
                         reconstruction=float(rloss.detach()),kl=float(kl.detach()),
                         generator_adversarial=float(adv.detach()),
                         discriminator=float(dloss.detach()),generator_total=float(gloss.detach()))
                history.append(rec)
                if global_step==1 or global_step % args.monitor_every==0:
                    save_probe_grid(probe,enc,dec,device,mon_dir/f'probe_step_{global_step:06d}.png')
                    save_metrics(history,mon_dir/'batch_losses.csv')
                    save_loss_plot(history,mon_dir/'batch_losses.png')
                    print(f'[monitor step={global_step}] rec={rec["reconstruction"]:.5f} '
                          f'KL={rec["kl"]:.5f} D={rec["discriminator"]:.5f} '
                          f'G={rec["generator_total"]:.5f} | {mon_dir}',flush=True)
        enc.eval();dec.eval();vals=[]
        with torch.inference_mode():
            for x in va:
                x=x.to(device);mu,_=enc(x);recon=dec(mu)
                vals.append((F.mse_loss(recon,x,reduction='none').mean((1,2,3))).cpu())
        v=float(torch.cat(vals).mean());t=float(np.mean(running))
        print(f'[epoch {epoch:03d}] train_rec={t:.6f} val_mse={v:.6f}')
        if args.monitor_every:
            save_probe_grid(probe,enc,dec,device,mon_dir/f'probe_epoch_{epoch:03d}.png')
            save_metrics(history,mon_dir/'batch_losses.csv')
            save_loss_plot(history,mon_dir/'batch_losses.png')
        if v<best:
            best=v
            torch.save({'encoder_state_dict':enc.state_dict(),'decoder_state_dict':dec.state_dict(),
                'discriminator_state_dict':disc.state_dict(),'latent_dims':args.latent_dims,
                'epoch':epoch,'val_mse':v,'config':vars(args),'model_kind':'r2ad_rectangles'},out)
            print('[saved]',out)
    print('[done] best validation MSE',best)
if __name__=='__main__':main()
