#!/usr/bin/env python3
"""Animate random augmented training patches and checkpoint reconstructions."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import load_models,read_mask
from r2ad.dataset import sample_training_patch,add_augmentation_arguments,augmentation_from_args,validate_augmentation
from r2ad.frame_manifest import read_frame_manifest
from scripts.animate_r2ad_inference import render_step


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--train-csv',required=True,type=Path)
    p.add_argument('--image',type=Path,help='Optional single image; must belong to normal train.csv')
    p.add_argument('--checkpoint','--checkpoints',nargs='+',required=True,type=Path,dest='checkpoints')
    p.add_argument('--output',type=Path,default=Path('reports/r2ad_training.gif'))
    p.add_argument('--device',default='auto')
    p.add_argument('--patch-size',type=int,default=224)
    p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--num-patches',type=int,default=24)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--augment',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--augmentation-mode',choices=['crop','full'],default='crop')
    add_augmentation_arguments(p)
    p.add_argument('--fps',type=float,default=3.)
    p.add_argument('--map-vmin',type=float,default=0.)
    p.add_argument('--map-vmax',type=float,default=.5)
    p.add_argument('--width',type=int,default=1024)
    p.add_argument('--safety-area',choices=['SafetyZone','PLeft','PRight','RoboArm','ConvBelt'],default='SafetyZone')
    args=p.parse_args()
    try:augmentation_settings=augmentation_from_args(args)
    except ValueError as exc:p.error(str(exc))
    supplied={token.split('=')[0].lstrip('-').replace('-','_') for token in sys.argv[1:] if token.startswith('--')}
    if args.num_patches<1 or args.patch_size<1 or args.width<720 or args.width%2 or args.fps<=0:
        p.error('Require positive patch counts/sizes/FPS and even width >=720')
    if not 0<args.min_coverage<=1 or not 0<=args.map_vmin<args.map_vmax<=1 or args.output.suffix.lower()!='.gif':
        p.error('Invalid coverage/color limits or output extension (must be .gif)')
    rows=read_frame_manifest(args.train_csv,with_metadata=True)
    if not rows or any(r.get('split')!='train' or r.get('label')!='normal' for r in rows):
        raise ValueError('Use a nonempty normal-only training manifest')
    if args.image:
        rows=[r for r in rows if r['source']==args.image.expanduser().resolve()]
        if not rows:
            raise ValueError('--image is not a member of the normal training split')
    cfg=load_config(args.config)
    value=cfg['r2ad']['safety_zone_mask'] if args.safety_area=='SafetyZone' else cfg['data']['mask_types'][args.safety_area]
    mask_path=Path(value).expanduser()
    if not mask_path.is_absolute():
        mask_path=Path(cfg['data']['dataset_base']).expanduser()/mask_path
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    choices=np.random.default_rng(args.seed).integers(len(rows),size=args.num_patches)
    frames=[];records=[]
    for checkpoint in args.checkpoints:
        enc,dec,ck=load_models(checkpoint,device)
        settings=validate_augmentation({key:ck.get('config',{}).get(key,value) if key not in supplied else value
                                       for key,value in augmentation_settings.items()})
        if ck.get('config',{}).get('safety_area',args.safety_area)!=args.safety_area:
            raise ValueError('Checkpoint and mask safety areas differ')
        for index,choice in enumerate(choices):
            row=rows[int(choice)]
            frame=cv2.imread(str(row['source']))
            if frame is None:
                raise ValueError(f'Unreadable training image: {row["source"]}')
            mask=read_mask(mask_path,frame.shape)
            patch,info=sample_training_patch(frame,mask,args.patch_size,args.min_coverage,
                np.random.default_rng(args.seed+index+1),args.augment,args.augmentation_mode,details=True,settings=settings)
            with torch.inference_mode():
                x=patch.unsqueeze(0).to(device)
                mu,_=enc(x);prediction=dec(mu)
                error=(x-prediction).abs().mean(1)[0].cpu().numpy()/2
                rgb=((prediction[0].permute(1,2,0).cpu().numpy()+1)/2).clip(0,1)
            bx,by,bw,bh=info['box']
            amap=np.zeros(mask.shape,np.float32)
            amap[by:by+bh,bx:bx+bw]=cv2.resize(error,(bw,bh),interpolation=cv2.INTER_LINEAR)
            covered=np.zeros(mask.shape,bool);covered[by:by+bh,bx:bx+bw]=True
            covered&=info['mask']>0
            label=f'Epoch {ck.get("epoch", "?")} | random patch {index+1}/{args.num_patches} | {row.get("scenario",row["source"].parent.name)} / {row["source"].name}'
            state=dict(index=index+1,total=args.num_patches,box=info['box'],map=amap,covered=covered,
                       input_rgb=((patch.permute(1,2,0).numpy()+1)/2).clip(0,1),reconstruction_rgb=rgb,patch_error=error,
                       scene_title='Augmented normal training scene; yellow=sample',
                       map_title=f'Current patch error only [{args.map_vmin:g}, {args.map_vmax:g}]',
                       status_lines=[label,'Random sampling + augmentation from trainer. Frozen checkpoint; no optimizer updates or map stitching.'])
            frames.append(render_step(info['scene'],info['mask'],state,[],args.width,args.map_vmin,args.map_vmax))
            records.append(dict(checkpoint=str(checkpoint.resolve()),epoch=ck.get('epoch'),sample=index+1,
                                source=str(row['source']),box=list(info['box']),augmentation=info['augmentation'],
                                augmentation_settings=settings,
                                mean_patch_error=float(error.mean())))
            print(label,flush=True)
        del enc,dec,ck
    args.output.parent.mkdir(parents=True,exist_ok=True)
    durations=[max(10,round(1000/args.fps))]*len(frames)
    for i in range(args.num_patches-1,len(frames),args.num_patches):
        durations[i]=2000
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    frames[-1].save(args.output.with_suffix('.png'))
    report=dict(samples=records,settings={k:[str(p) for p in v] if k=='checkpoints' else str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                note='Uses the actual image-training sampler and augmentations. Frozen latent-mean reconstructions; not a replay of optimizer steps. Multiple checkpoints reuse identical samples and augmentations in supplied order. No training anomaly map is stitched.')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {args.output}',flush=True)


if __name__=='__main__':
    main()
