#!/usr/bin/env python3
"""Compare checkpoints on the same diverse, labeled test frames."""
import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import load_models, read_mask, infer_map, save_map
from r2ad.frame_manifest import read_frame_manifest
from scripts.preview_r2ad_splits import sample_rows, scenario


def choose_checkpoints(directory, count):
    best=directory/'model_r2ad.pt'
    periodic=sorted((directory/'checkpoints').glob('*.pt'))
    if best.is_file():
        count-=1
    selected=[periodic[i] for i in np.linspace(0,len(periodic)-1,min(max(count,0),len(periodic)),dtype=int)] if periodic and count else []
    return ([best] if best.is_file() else [])+selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--split-dir',required=True,type=Path)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint-dir',type=Path,help='Directory containing model_r2ad.pt and/or checkpoints/*.pt')
    group.add_argument('--checkpoints',nargs='+',type=Path,help='Explicit checkpoint files')
    p.add_argument('--max-checkpoints',type=int,default=4,help='Directory mode: best plus evenly spaced periodic checkpoints')
    p.add_argument('--samples-per-class',type=int,default=6)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    p.add_argument('--safety-area',default='SafetyZone',choices=['SafetyZone','PLeft','PRight','RoboArm','ConvBelt'])
    p.add_argument('--patch-size',type=int,default=224)
    p.add_argument('--stride',type=int,default=64)
    p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--sigma',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--map-vmin',type=float,default=0.)
    p.add_argument('--map-vmax',type=float,default=.5)
    p.add_argument('--output-dir',type=Path,default=Path('reports/r2ad_checkpoint_inspection'))
    args=p.parse_args()
    if min(args.max_checkpoints,args.samples_per_class,args.patch_size,args.stride,args.batch_size)<1 or not 0<args.min_coverage<=1 or args.sigma<0:
        p.error('Require positive counts, patch/stride/batch sizes, 0 < min-coverage <= 1, sigma >= 0')
    if not 0<=args.map_vmin<args.map_vmax<=1:
        p.error('Require 0 <= map-vmin < map-vmax <= 1')
    checkpoints=args.checkpoints or choose_checkpoints(args.checkpoint_dir,args.max_checkpoints)
    if not checkpoints:
        p.error('No checkpoints found; point --checkpoint-dir to the directory containing model_r2ad.pt')
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    cfg=load_config(args.config)
    mask_value=cfg['r2ad']['safety_zone_mask'] if args.safety_area=='SafetyZone' else cfg['data']['mask_types'][args.safety_area]
    mask_path=Path(mask_value).expanduser()
    if not mask_path.is_absolute():
        mask_path=Path(cfg['data']['dataset_base']).expanduser()/mask_path
    rows=read_frame_manifest(args.split_dir/'test.csv',with_metadata=True)
    if any(row.get('split')!='test' for row in rows):
        raise ValueError('Expected test-only rows')
    selected=[]
    for label in ('normal','anomalous'):
        candidates=[row for row in rows if row.get('label' if args.safety_area=='SafetyZone' else f'label_{args.safety_area}')==label]
        if not candidates:
            raise ValueError(f'No {label} test examples for {args.safety_area}')
        selected+=sample_rows(candidates,args.samples_per_class,args.seed)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    selection=[{k:str(v) if isinstance(v,Path) else v for k,v in row.items()} for row in selected]
    (args.output_dir/'selected_frames.json').write_text(json.dumps(selection,indent=2))
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    records=[];sections=[]
    for ci,checkpoint in enumerate(checkpoints):
        enc,dec,ck=load_models(checkpoint,device)
        settings=ck.get('config',{})
        if settings.get('safety_area',args.safety_area)!=args.safety_area:
            raise ValueError(f'{checkpoint} was trained for {settings["safety_area"]}, not {args.safety_area}')
        folder=args.output_dir/f'{ci:02d}_{checkpoint.stem}'
        folder.mkdir(parents=True,exist_ok=True)
        cards=[]
        for index,row in enumerate(selected):
            frame=cv2.imread(str(row['source']))
            if frame is None:
                raise ValueError(f'Unreadable frame: {row["source"]}')
            mask=read_mask(mask_path,frame.shape)
            result=infer_map(frame,mask,enc,dec,device,args.patch_size,args.stride,args.min_coverage,
                             args.batch_size,args.sigma,return_reconstruction=True)
            prefix=f'{index:02d}_{scenario(row)}_{row.get("frame_id",row["source"].stem)}'
            save_map(folder/f'{prefix}_overlay.png',frame,mask,result,args.map_vmin,args.map_vmax)
            heat=cv2.imread(str(folder/f'{prefix}_overlay_heat.png'))
            # Gray marks mask pixels that the inference grid could not cover.
            reconstruction=result['reconstruction'].copy()
            reconstruction[(mask>0)&~result['covered']]=(128,128,128)
            cv2.imwrite(str(folder/f'{prefix}_reconstruction.png'),reconstruction)
            original=np.where((mask>0)[...,None],frame,0)
            heat[~result['covered']]=(128,128,128)
            panels=[]
            for image,title in [(frame,'Original scene'),(original,'Original SafetyZone/area'),
                                (reconstruction,'Stitched reconstruction (gray = uncovered)'),
                                (heat,f'Reds residual [{args.map_vmin:g}, {args.map_vmax:g}]; gray = unscored')]:
                width=720;height=round(image.shape[0]*width/image.shape[1])
                panel=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
                panel=cv2.copyMakeBorder(panel,32,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
                cv2.putText(panel,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                panels.append(panel)
            sheet=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
            filename=f'{prefix}_comparison.jpg'
            if not cv2.imwrite(str(folder/filename),sheet):
                raise IOError(filename)
            label=row['label' if args.safety_area=='SafetyZone' else f'label_{args.safety_area}']
            rec=dict(checkpoint=str(checkpoint.resolve()),epoch=ck.get('epoch'),source=str(row['source']),
                     scenario=scenario(row),frame_id=row.get('frame_id',''),label=label,score=result['score'],
                     mean_error=float(result['map'][result['covered']].mean()),covered_mask_fraction=result['coverage_ratio'])
            rec.update({f'label_{a}':row.get(f'label_{a}','unknown') for a in ('PLeft','PRight','RoboArm','ConvBelt')})
            records.append(rec)
            description=f'{label} | scenario={rec["scenario"]} frame={rec["frame_id"]} | score={rec["score"]:.5f} coverage={rec["covered_mask_fraction"]:.1%}'
            areas=' | '.join(f'{a}={rec[f"label_{a}"]}' for a in ('PLeft','PRight','RoboArm','ConvBelt'))
            relative=f'{folder.name}/{filename}'
            cards.append(f'<h3>{html.escape(description)}</h3><p>{html.escape(areas)}</p><a href="{relative}"><img width="100%" src="{relative}"></a>')
            print(f'[{ci+1}/{len(checkpoints)}] {checkpoint.name} | {description}',flush=True)
        sections.append(f'<h2>{html.escape(checkpoint.name)} | epoch {ck.get("epoch","?")}</h2>'+''.join(cards))
        del enc,dec,ck
    with (args.output_dir/'scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    class_summary=[]
    for checkpoint in dict.fromkeys(row['checkpoint'] for row in records):
        for label in ('normal','anomalous'):
            subset=[row for row in records if row['checkpoint']==checkpoint and row['label']==label]
            class_summary.append(dict(checkpoint=checkpoint,label=label,n=len(subset),
                mean_score=float(np.mean([r['score'] for r in subset])),
                median_score=float(np.median([r['score'] for r in subset])),
                mean_reconstruction_error=float(np.mean([r['mean_error'] for r in subset])),
                mean_covered_mask_fraction=float(np.mean([r['covered_mask_fraction'] for r in subset]))))
    with (args.output_dir/'class_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(class_summary[0]));writer.writeheader();writer.writerows(class_summary)
    report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='checkpoints'},
                checkpoints=[str(c.resolve()) for c in checkpoints],sample_count=len(selected),class_summary=class_summary,
                note='Exploratory test subset; do not select/tune checkpoints on test scores. Black reconstruction outside covered mask is not a model prediction.')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    (args.output_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2AD checkpoint inspection</title>'
        '<h1>Checkpoint reconstruction inspection</h1><p>'+html.escape(report['note'])+'</p>'+''.join(sections))
    print(f'Open {args.output_dir / "index.html"}',flush=True)


if __name__=='__main__':
    main()
