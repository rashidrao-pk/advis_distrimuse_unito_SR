#!/usr/bin/env python3
"""Audit an existing encoder on reproducible unaugmented normal patches."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import load_models,images,read_mask,extract,random_boxes
from r2ad.frame_manifest import read_frame_manifest
from r2ad.latent_audit import save_latent_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path,help='Normal val.csv; exploratory image/directory requires explicit flag')
    p.add_argument('--allow-unverified-input',action='store_true')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--samples',type=int,default=1024)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--active-threshold',type=float,default=.01)
    p.add_argument('--save-codes',action='store_true')
    p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    if args.samples<2 or args.batch_size<1 or args.cpu_threads<1 or args.active_threshold<0:
        p.error('Require samples >=2, positive batch/CPU threads, nonnegative active threshold')
    torch.set_num_threads(args.cpu_threads);cv2.setNumThreads(1)
    if args.input.suffix.lower()=='.csv':
        rows=read_frame_manifest(args.input,with_metadata=True)
        if any(r.get('label')!='normal' or r.get('split')!='val' for r in rows):
            raise ValueError('Use normal-only validation rows, not anomalous/test or training frames')
        paths=[r['source'] for r in rows];scope='held_out_normal_validation'
    elif args.allow_unverified_input:
        paths=images(args.input);scope='exploratory_unverified_input_not_capacity_evidence'
    else:p.error('Use val.csv, or explicitly allow an exploratory unverified image/directory')
    if not paths:raise ValueError('No normal frames available')
    cfg=load_config(args.config)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    enc,dec,ck=load_models(args.checkpoint,device);del dec
    settings=ck.get('config',{});patch=int(settings.get('patch_size',224))
    area=settings.get('safety_area','SafetyZone')
    value=cfg['r2ad']['safety_zone_mask'] if area=='SafetyZone' else cfg['data']['mask_types'][area]
    mask_path=Path(value).expanduser()
    if not mask_path.is_absolute():mask_path=Path(cfg['data']['dataset_base']).expanduser()/mask_path
    # Match the checkpoint's unaugmented validation crop-size distribution.
    scale=(1.,1.) if settings.get('fixed_patch_size') else (.9,1.1)
    rng=np.random.default_rng(args.seed)
    order=rng.permutation(len(paths));selected=[paths[int(order[i%len(order)])] for i in range(args.samples)]
    mus=[];logvars=[];pending=[];used=set();last_path=None
    def encode_pending():
        if not pending:return
        x=torch.stack(pending).to(device)
        with torch.inference_mode():
            mu,logvar=enc(x)
        mus.append(mu.float().cpu().numpy());logvars.append(logvar.float().cpu().numpy());pending.clear()
    for index,path in enumerate(selected):
        if path!=last_path:
            frame=cv2.imread(str(path))
            if frame is None:raise ValueError(f'Unreadable frame: {path}')
            mask=read_mask(mask_path,frame.shape);last_path=path
        box=random_boxes(mask,patch,1,rng,float(settings.get('min_coverage',.85)),scale=scale)[0]
        pending.append(extract(frame,box));used.add(str(path))
        if len(pending)==args.batch_size:encode_pending()
    encode_pending()
    summary=save_latent_audit(np.concatenate(mus),np.concatenate(logvars),args.output_dir,
        dict(checkpoint=str(args.checkpoint.resolve()),epoch=ck.get('epoch'),input=str(args.input.resolve()),scope=scope,
             unique_frames=len(used),unique_recording_groups=len({str(Path(path).parent) for path in used}),
             patch_size=patch,patch_scale=scale,seed=args.seed,device=device,precision='fp32'),args.active_threshold,args.save_codes)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
