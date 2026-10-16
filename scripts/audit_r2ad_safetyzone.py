#!/usr/bin/env python3
"""Audit R2AD SafetyZone training coverage and a held-out validation frame.

Run from repo root: python scripts/audit_r2ad_safetyzone.py --config ...
Requires existing r2ad.core, r2ad.mcap_data, and r2ad.config modules.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.core import read_mask, random_boxes, boxes, extract, load_models
from r2ad.mcap_data import McapFrameReader, load_manifest
from r2ad.config import load_config

AREAS = ('PLeft', 'PRight', 'RoboArm', 'ConvBelt')
COLORS = [(240,140,20),(20,180,240),(30,220,70),(200,90,220)]


def path_from(root, entry):
    p=Path(entry).expanduser()
    return p if p.is_absolute() else root/p


def overlay_boxes(image, coords, color=(0,220,255), number=True):
    vis=image.copy()
    for j,(x,y,w,h) in enumerate(coords):
        cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
        if number:
            cv2.putText(vis,str(j+1),(x+4,y+22),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    return vis


def patch_coverage(mask, coords):
    hits=np.zeros(mask.shape,np.uint16)
    for x,y,w,h in coords:
        hits[y:y+h,x:x+w] += 1
    return hits


def stats_for(hits, masks):
    out={}
    for name,mask in masks.items():
        positive=mask>0
        total=int(positive.sum())
        out[name]={
            'mask_pixels':total,
            'covered_pixels':int(np.count_nonzero((hits>0)&positive)),
            'coverage_percent':round(100*np.count_nonzero((hits>0)&positive)/max(total,1),2),
            'mean_exposures':round(float(hits[positive].mean()),3) if total else 0.,
            'never_seen_pixels':int(np.count_nonzero((hits==0)&positive))
        }
    return out


def save_heatmap(hits, mask, filepath):
    v=np.clip(hits.astype(np.float32)/max(float(np.quantile(hits[mask>0],.99)),1),0,1)
    cm=cv2.applyColorMap((v*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    cm[mask==0]=(30,30,30)
    cv2.imwrite(str(filepath),cm)


def save_reconstruction_grid(frame, coords, checkpoint, out, device, n=6):
    enc,dec,ck=load_models(checkpoint,device)
    rows=[]
    for box in coords[:n]:
        x=extract(frame,box).unsqueeze(0).to(device)
        with torch.inference_mode():
            mu,_=enc(x)
            pred=dec(mu)
        orig=((x[0].permute(1,2,0).cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        reco=((pred[0].permute(1,2,0).cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
        err=np.abs(orig.astype(np.float32)-reco.astype(np.float32)).mean(axis=2)
        v=max(float(np.quantile(err,.99)),1.)
        heat=cv2.applyColorMap(np.uint8(np.clip(err/v,0,1)*255),cv2.COLORMAP_TURBO)
        rgb=np.concatenate([orig,reco,heat],axis=1)
        rows.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if rows:
        header=np.full((34,384,3),32,np.uint8)
        for txt,x in [('INPUT',20),('RECONSTRUCTION',140),('ABS ERROR',275)]:
            cv2.putText(header,txt,(x,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.imwrite(str(out),np.concatenate([header,*rows],axis=0))


def main():
    p=argparse.ArgumentParser(description='Visual audit of randomized training patches and fixed inference grid')
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--manifest',default=None)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--frame-index',type=int,default=0)
    p.add_argument('--patch-size',type=int,default=224)
    p.add_argument('--stride',type=int,default=64)
    p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--samples-per-frame',type=int,default=6)
    p.add_argument('--simulate-frames',type=int,default=50,
                   help='Count random sampling opportunities from a given mask across this many simulated frame draws')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--output-dir',default=None)
    args=p.parse_args()
    cfg=load_config(args.config)
    root=Path(cfg['data']['dataset_base']).expanduser()
    rcfg=cfg.get('r2ad') or {}
    manifest=path_from(root,args.manifest or rcfg.get('manifest') or 'r2ad/manifest.csv')
    rows=load_manifest(manifest,args.split)
    if args.frame_index<0 or args.frame_index>=len(rows):
        p.error(f'--frame-index must be from 0 to {len(rows)-1}')
    out=Path(args.output_dir) if args.output_dir else Path('results')/'r2ad'/'coverage_audit'
    out.mkdir(parents=True,exist_ok=True)
    reader=McapFrameReader()
    try: frame=reader.read(rows[args.frame_index])
    finally: reader.close()
    hh,ww=frame.shape[:2]
    mask_paths=cfg['data']['mask_types']
    masks={name:read_mask(path_from(root,mask_paths[name]),frame.shape) for name in AREAS}
    union=np.logical_or.reduce([m.astype(bool) for m in masks.values()]).astype(np.uint8)
    existing=rcfg.get('safety_zone_mask') or mask_paths.get('SafetyZone')
    if existing:
        disk_mask=read_mask(path_from(root,existing),frame.shape)
        if not np.array_equal(union,disk_mask):
            print('[WARNING] The saved SafetyZone mask does not match the OR of the four masks.')
    rng=np.random.default_rng(args.seed)
    six=random_boxes(union,args.patch_size,args.samples_per_frame,rng,args.min_coverage)
    random_many=[]
    for _ in range(args.simulate_frames):
        random_many.extend(random_boxes(union,args.patch_size,args.samples_per_frame,rng,args.min_coverage))
    try: fixed=boxes(union,args.patch_size,args.stride,args.min_coverage)
    except ValueError as e:
        print('[WARNING] Fixed grid has no valid boxes:',e)
        fixed=[]
    summary={
        'input':{'scenario':rows[args.frame_index]['scenario'],'frame_shape':[hh,ww],
                 'split':args.split,'manifest_records':len(rows)},
        'six_random_rectangles':stats_for(patch_coverage(union,six),{'SafetyZone':union,**masks}),
        'simulation':{'frames':args.simulate_frames,'patches':len(random_many),
                      'coverage':stats_for(patch_coverage(union,random_many),{'SafetyZone':union,**masks})},
        'fixed_inference_grid':{'rectangles':len(fixed),
                                'coverage':stats_for(patch_coverage(union,fixed),{'SafetyZone':union,**masks})},
        'note':'Simulation is a diagnostic random draw on this mask; it is not a record of rectangles actually seen during training.'
    }
    # Safety regions visualization
    region=frame.copy()
    for (name,m),col in zip(masks.items(),COLORS):
        tint=np.zeros_like(frame);tint[:]=col
        sel=m>0
        region[sel]=cv2.addWeighted(frame,.65,tint,.35,0)[sel]
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(region,cnts,-1,col,2)
    cv2.imwrite(str(out/'01_safetyzone_areas.png'),region)
    cv2.imwrite(str(out/'02_random_6_rectangles.png'),overlay_boxes(region,six))
    save_heatmap(patch_coverage(union,random_many),union,out/'03_random_sampling_frequency.png')
    cv2.imwrite(str(out/'04_fixed_grid.png'),overlay_boxes(region,fixed,number=False))
    save_heatmap(patch_coverage(union,fixed),union,out/'05_fixed_grid_coverage.png')
    if args.checkpoint:
        device='cuda' if torch.cuda.is_available() else 'cpu'
        save_reconstruction_grid(frame,six,args.checkpoint,out/'06_reconstructions.png',device)
    (out/'coverage_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('[saved]',out)

if __name__=='__main__':main()
