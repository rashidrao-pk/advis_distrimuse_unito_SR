#!/usr/bin/env python3
"""Compare source patch counts and exact pixel coverage for an existing ROI mask."""
import argparse
import csv
import html
import json
import math
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import boxes,random_boxes,read_mask

AREAS=('PLeft','PRight','RoboArm','ConvBelt')


def coverage_counts(rectangles,shape):
    delta=np.zeros((shape[0]+1,shape[1]+1),np.int32)
    if rectangles:
        values=np.asarray(rectangles);x,y,w,h=values.T
        for yy,xx,value in [(y,x,1),(y+h,x,-1),(y,x+w,-1),(y+h,x+w,1)]:
            np.add.at(delta,(yy,xx),value)
    return delta.cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)[:-1,:-1]


def grid_estimate(mask,areas,patch,stride,min_coverage,target):
    try:
        rectangles=boxes(mask,patch,stride,min_coverage)
    except ValueError:
        rectangles=[]
    counts=coverage_counts(rectangles,mask.shape)
    covered=counts>0;selected=mask>0
    fraction=float(covered[selected].mean()) if selected.any() else 0.
    row=dict(patch_size=patch,model_size=128,stride=stride,patch_count=len(rectangles),
             optimistic_area_lower_bound=math.ceil(int(selected.sum())/(patch*patch)),
             safety_zone_coverage=fraction,uncovered_safety_pixels=int((selected&~covered).sum()),
             mean_contributions_per_safety_pixel=float(counts[selected].mean()) if selected.any() else 0.,
             meets_target=bool(fraction>=target))
    for name,area in areas.items():
        row[name+'_coverage']=float(covered[area>0].mean()) if np.any(area) else None
    return row,rectangles,covered


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--image',type=Path,help='Optional original frame for grid overlays')
    p.add_argument('--patch-sizes',nargs='+',type=int,default=[32,48,64,96,128,224])
    p.add_argument('--stride-ratios',nargs='+',type=float,default=[1.,.5,.25])
    p.add_argument('--min-coverage',type=float,default=.85,help='Required fraction of each patch inside SafetyZone')
    p.add_argument('--target-coverage',type=float,default=.99,help='Target fraction of all SafetyZone pixels covered')
    p.add_argument('--random-budgets',nargs='+',type=int,default=[6,24,48,96,192])
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--strict-area-masks',action='store_true',help='Fail if any component area mask is unavailable; otherwise report combined coverage')
    p.add_argument('--output-dir',type=Path,default=Path('reports/r2ad_patch_coverage'))
    args=p.parse_args()
    if min(args.patch_sizes)<16 or min(args.stride_ratios)<=0 or max(args.stride_ratios)>1 or min(args.random_budgets)<1 or args.trials<1:
        p.error('Require source patches >=16, stride ratios in (0,1], positive random budgets/trials')
    if not 0<args.min_coverage<=1 or not 0<args.target_coverage<=1:
        p.error('Coverage fractions must be in (0,1]')
    cfg=load_config(args.config);root=Path(cfg['data']['dataset_base']).expanduser()
    def path(value):
        value=Path(value).expanduser();return value if value.is_absolute() else root/value
    mask_path=path(cfg['r2ad']['safety_zone_mask'])
    raw=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if raw is None:raise FileNotFoundError(mask_path)
    mask=(raw>127).astype(np.uint8)
    if not mask.any():raise ValueError('SafetyZone mask is empty')
    areas={};missing_areas=[]
    for name in AREAS:
        value=cfg['data'].get('mask_types',{}).get(name)
        if not value or not path(value).is_file():
            if args.strict_area_masks:raise FileNotFoundError(f'Missing {name} mask: {value}')
            missing_areas.append(name)
            print(f'[area] {name} mask unavailable; combined SafetyZone estimates remain valid',flush=True)
            continue
        areas[name]=read_mask(path(value),mask.shape)
    frame=cv2.imread(str(args.image)) if args.image else np.full((*mask.shape,3),90,np.uint8)
    if frame is None or frame.shape[:2]!=mask.shape:raise ValueError('Image must match original mask resolution')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    grid_rows=[];random_rows=[];links=[]
    print('Source size | stride | patches | SafetyZone coverage | optimistic area bound',flush=True)
    for patch in sorted(set(args.patch_sizes)):
        if patch>min(mask.shape):raise ValueError(f'Patch {patch} exceeds image dimensions')
        for stride in sorted({max(1,round(patch*r)) for r in args.stride_ratios},reverse=True):
            row,rectangles,covered=grid_estimate(mask,areas,patch,stride,args.min_coverage,args.target_coverage)
            grid_rows.append(row)
            visual=(frame.astype(float)*.35).astype(np.uint8)
            visual[(mask>0)&covered]=(.65*frame[(mask>0)&covered]+.35*np.array([0,255,0])).astype(np.uint8)
            visual[(mask>0)&~covered]=(0,0,255)
            for x,y,w,h in rectangles:cv2.rectangle(visual,(x,y),(x+w-1,y+h-1),(0,210,255),1)
            visual=cv2.copyMakeBorder(visual,58,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            text=f'{patch}x{patch} -> 128x128 model | stride={stride} | {len(rectangles)} patches | coverage={row["safety_zone_coverage"]:.2%}'
            cv2.putText(visual,text,(12,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
            name=f'p{patch}_s{stride}.jpg'
            if not cv2.imwrite(str(args.output_dir/name),visual):raise IOError(name)
            links.append(f'<h3>{html.escape(text)}</h3><img style="max-width:100%" src="{name}">')
            print(f'{patch:>11} | {stride:>6} | {len(rectangles):>7} | {row["safety_zone_coverage"]:>18.2%} | {row["optimistic_area_lower_bound"]}',flush=True)
        trials={budget:[] for budget in sorted(set(args.random_budgets))}
        for trial in range(args.trials):
            try:
                rectangles=random_boxes(mask,patch,max(trials),np.random.default_rng(args.seed+trial),args.min_coverage,scale=(1.,1.))
            except ValueError:
                break
            covered=np.zeros(mask.shape,bool)
            for index,(x,y,w,h) in enumerate(rectangles,1):
                covered[y:y+h,x:x+w]=True
                if index in trials:trials[index].append(float(covered[mask>0].mean()))
        for budget,values in trials.items():
            random_rows.append(dict(patch_size=patch,random_patches=budget,trials=len(values),
                                    mean_safety_coverage=float(np.mean(values)) if values else None,
                                    min_safety_coverage=float(min(values)) if values else None,
                                    max_safety_coverage=float(max(values)) if values else None))
    for name,rows in [('grid_coverage.csv',grid_rows),('random_coverage.csv',random_rows)]:
        with (args.output_dir/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(mask=str(mask_path.resolve()),mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                mask_pixels=int(mask.sum()),image_shape=list(mask.shape),seed=args.seed,
                min_patch_occupancy=args.min_coverage,target_mask_coverage=args.target_coverage,
                grids=grid_rows,random_sampling=random_rows,unavailable_area_masks=missing_areas,
                note='Exact counts for the evaluation grid, not optimal packing. End-aligned grid windows can overlap even at stride=patch. Area bound assumes no wasted pixels and is not achievable in general. Random trials use one unaugmented frame, fixed crop size, no zoom. Smaller patches do not remove spatial/temporal correlations.')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    table='<table border="1" cellpadding="6"><tr><th>Source size</th><th>Stride</th><th>Grid patches</th><th>Mask coverage</th><th>Target met</th></tr>'
    for row in grid_rows:
        table+=f'<tr><td>{row["patch_size"]}</td><td>{row["stride"]}</td><td>{row["patch_count"]}</td><td>{row["safety_zone_coverage"]:.2%}</td><td>{row["meets_target"]}</td></tr>'
    table+='</table>'
    (args.output_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Patch coverage</title><h1>Source patch coverage</h1>'
        '<p>Green: covered mask; red: uncovered mask; yellow: accepted patch boundaries. Crops resize to 128x128.</p><p>'
        +html.escape(report['note'])+'</p><p>Unavailable area masks: '+html.escape(', '.join(missing_areas) or 'none')+'</p>'+table+''.join(links))
    print(f'Open {args.output_dir/"index.html"}',flush=True)


if __name__=='__main__':main()
