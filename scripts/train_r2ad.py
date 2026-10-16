#!/usr/bin/env python3
import argparse,sys,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.core import build_models
from r2ad.dataset import RectangleDataset

def main():
    p=argparse.ArgumentParser(description='Train R2AD VAE-GAN using normal full-resolution images')
    for flag in ('train-dir','val-dir','mask','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--patch-size',type=int,default=224);p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--samples-per-frame',type=int,default=6);p.add_argument('--latent-dims',type=int,default=64)
    p.add_argument('--lr',type=float,default=.0002);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--kl-weight',type=float,default=.0001);p.add_argument('--adv-weight',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='auto')
    args=p.parse_args();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    train=RectangleDataset(args.train_dir,args.mask,args.patch_size,args.min_coverage,args.samples_per_frame,True,args.seed)
    val=RectangleDataset(args.val_dir,args.mask,args.patch_size,args.min_coverage,1,False,args.seed)
    tr=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device=='cuda')
    va=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device=='cuda')
    enc,dec,disc=build_models(args.latent_dims,device)
    opt_g=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=args.lr)
    opt_d=torch.optim.Adam(disc.parameters(),lr=args.lr/2)
    bce=nn.BCEWithLogitsLoss();best=float('inf');out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    print(f'[data] train frames={len(train.paths)}, val frames={len(val.paths)}, train patches/epoch={len(train)}')
    for epoch in range(1,args.epochs+1):
        enc.train();dec.train();disc.train();running=[]
        for x in tr:
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
        enc.eval();dec.eval();vals=[]
        with torch.inference_mode():
            for x in va:
                x=x.to(device);mu,_=enc(x);recon=dec(mu)
                vals.append((F.mse_loss(recon,x,reduction='none').mean((1,2,3))).cpu())
        v=float(torch.cat(vals).mean());t=float(np.mean(running))
        print(f'[epoch {epoch:03d}] train_rec={t:.6f} val_mse={v:.6f}')
        if v<best:
            best=v
            torch.save({'encoder_state_dict':enc.state_dict(),'decoder_state_dict':dec.state_dict(),
                'discriminator_state_dict':disc.state_dict(),'latent_dims':args.latent_dims,
                'epoch':epoch,'val_mse':v,'config':vars(args),'model_kind':'r2ad_rectangles'},out)
            print('[saved]',out)
    print('[done] best validation MSE',best)
if __name__=='__main__':main()
