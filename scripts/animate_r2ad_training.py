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
from r2ad.core import load_models,read_mask,checkpoint_patch_settings
from r2ad.dataset import sample_training_patch,add_augmentation_arguments,augmentation_from_args,validate_augmentation
from r2ad.frame_manifest import read_frame_manifest
from scripts.animate_r2ad_inference import render_step
from r2ad.frame_preview import full_frame_preview,append_panels,AREAS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--train-csv',required=True,type=Path)
    p.add_argument('--image',type=Path,help='Optional single image; must belong to normal train.csv')
    p.add_argument('--checkpoint','--checkpoints',nargs='+',required=True,type=Path,dest='checkpoints')
    p.add_argument('--output',type=Path,default=Path('reports/r2ad_training.gif'))
    p.add_argument('--device',default='auto')
    p.add_argument('--patch-size',type=int,default=None)
    p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--num-patches',type=int,default=24)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--augment',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--augmentation-mode',choices=['crop','full'],default='crop')
    add_augmentation_arguments(p)
    p.add_argument('--fps',type=float,default=3.)
    p.add_argument('--map-vmin',type=float,default=0.)
    p.add_argument('--map-vmax',type=float,default=.5)
    p.add_argument('--width',type=int,default=1024,help='Section width; landscape full-frame GIF is twice this width')
    p.add_argument('--layout',choices=['landscape','portrait'],default='landscape',help='Full-frame panels beside patch view (default) or below it')
    p.add_argument('--full-frame',action='store_true',help='Append stitched scene, combined SafetyZone, and per-area maps')
    p.add_argument('--full-frame-scope',choices=['safety'],default='safety',help='Only SafetyZone patches are reconstructed and scored')
    p.add_argument('--outside-dim',type=float,default=.35,help='Input brightness outside SafetyZone in reconstruction/overlay (0..1)')
    p.add_argument('--stride',type=int,default=None,help='Full-frame grid stride; defaults from checkpoint patch size')
    p.add_argument('--batch-size',type=int,default=32,help='Full-frame inference batch size')
    p.add_argument('--sigma',type=float,default=1.,help='Full-frame residual smoothing, same as evaluation')
    p.add_argument('--safety-area',choices=['SafetyZone','PLeft','PRight','RoboArm','ConvBelt'],default='SafetyZone')
    args=p.parse_args()
    try:augmentation_settings=augmentation_from_args(args)
    except ValueError as exc:p.error(str(exc))
    supplied={token.split('=')[0].lstrip('-').replace('-','_') for token in sys.argv[1:] if token.startswith('--')}
    if args.num_patches<1 or args.width<720 or args.width%2 or args.fps<=0:
        p.error('Require positive patch counts/sizes/FPS and even width >=720')
    if args.batch_size<1 or args.sigma<0 or not 0<=args.outside_dim<=1:p.error('Require batch-size >0, sigma >=0 and outside-dim in [0,1]')
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
    if args.full_frame:
        if args.safety_area!='SafetyZone':p.error('--full-frame combined/per-area view requires a SafetyZone model')
        missing=set(AREAS)-set(cfg.get('data',{}).get('mask_types',{}))
        if missing:p.error(f'--full-frame requires all four original area masks; missing {sorted(missing)}')
    value=cfg['r2ad']['safety_zone_mask'] if args.safety_area=='SafetyZone' else cfg['data']['mask_types'][args.safety_area]
    mask_path=Path(value).expanduser()
    if not mask_path.is_absolute():
        mask_path=Path(cfg['data']['dataset_base']).expanduser()/mask_path
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    choices=np.random.default_rng(args.seed).integers(len(rows),size=args.num_patches)
    frames=[];records=[];area_cache={}
    for checkpoint in args.checkpoints:
        enc,dec,ck=load_models(checkpoint,device)
        patch_size,stride=checkpoint_patch_settings(ck,args.patch_size,args.stride)
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
            patch,info=sample_training_patch(frame,mask,patch_size,args.min_coverage,
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
                       status_lines=[label,('Random patch left; full-grid diagnostic right. Frozen checkpoint, no optimizer updates.' if args.layout=='landscape'
                                           else 'Random patch above; full-grid diagnostic below. Frozen checkpoint, no optimizer updates.') if args.full_frame
                                     else 'Random sampling + augmentation from trainer. Frozen checkpoint; no optimizer updates or map stitching.'])
            visual=render_step(info['scene'],info['mask'],state,[],args.width,args.map_vmin,args.map_vmax)
            full_report=None
            if args.full_frame:
                transformed={}
                for name in AREAS:
                    value=Path(cfg['data']['mask_types'][name]).expanduser()
                    if not value.is_absolute():value=Path(cfg['data']['dataset_base']).expanduser()/value
                    key=(str(value),frame.shape[:2])
                    if key not in area_cache:area_cache[key]=read_mask(value,frame.shape)
                    transformed[name]=cv2.warpAffine(area_cache[key],info['affine_matrix'],(frame.shape[1],frame.shape[0]),
                                                     flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                extra,full_report=full_frame_preview(info['scene'],info['mask'],transformed,enc,dec,device,
                    patch_size,stride,args.min_coverage,args.batch_size,args.sigma,args.width,args.map_vmin,args.map_vmax,args.full_frame_scope,args.outside_dim)
                visual=append_panels(visual,extra,args.layout)
            frames.append(visual)
            records.append(dict(checkpoint=str(checkpoint.resolve()),epoch=ck.get('epoch'),sample=index+1,
                                source=str(row['source']),box=list(info['box']),augmentation=info['augmentation'],
                                augmentation_settings=settings,
                                mean_patch_error=float(error.mean()),full_frame=full_report))
            print(label,flush=True)
        del enc,dec,ck
    args.output.parent.mkdir(parents=True,exist_ok=True)
    durations=[max(10,round(1000/args.fps))]*len(frames)
    for i in range(args.num_patches-1,len(frames),args.num_patches):
        durations[i]=2000
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    frames[-1].save(args.output.with_suffix('.png'))
    report=dict(samples=records,settings={k:[str(p) for p in v] if k=='checkpoints' else str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                note='Uses the actual image-training sampler and augmentations. Frozen latent-mean reconstructions; not optimizer updates. Full-frame panels stitch only SafetyZone patches on the same augmented scene. Outside is faded input, not reconstruction, and white in maps. Gray marks uncovered ROI. All masks use the identical affine transform; area scores are uncalibrated views of one model.')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {args.output}',flush=True)


if __name__=='__main__':
    main()
