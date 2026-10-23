#!/usr/bin/env python3
"""Animate patch inference and Hann-weighted residual-map formation for one frame."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import load_models,read_mask,infer_map,save_map


def render_step(frame, mask, state, visited, width, vmin, vmax):
    """Render one explanatory frame; untouched mask pixels are gray, not zero error."""
    def red(values):
        rgb=colormaps['Reds'](np.clip((values-vmin)/(vmax-vmin),0,1),bytes=True)[...,:3]
        return np.ascontiguousarray(rgb[...,::-1])
    def panel(image,title,w,h):
        image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
        image=cv2.copyMakeBorder(image,30,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(image,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        return image
    scene=frame.copy()
    for x,y,w,h in visited:
        cv2.rectangle(scene,(x,y),(x+w-1,y+h-1),(0,160,0),1)
    x,y,w,h=state['box']
    cv2.rectangle(scene,(x,y),(x+w-1,y+h-1),(0,255,255),4)
    heat=red(state['map'])
    heat[mask==0]=(30,30,30)
    heat[(mask>0)&~state['covered']]=(140,140,140)
    half=width//2;scene_h=round(frame.shape[0]*half/frame.shape[1])
    top=np.hstack([panel(scene,'Grid: yellow=current, green=processed',half,scene_h),
                   panel(heat,f'Accumulated Reds map [{vmin:g}, {vmax:g}]',half,scene_h)])
    tile=width//3;tile_h=min(tile,224)
    inputs=(state['input_rgb']*255).astype(np.uint8)[...,::-1]
    reconstruction=(state['reconstruction_rgb']*255).astype(np.uint8)[...,::-1]
    bottom=np.hstack([panel(inputs,'Model input (128 x 128)',tile,tile_h),
                      panel(reconstruction,'Reconstruction (latent mean)',tile,tile_h),
                      panel(red(state['patch_error']),'Patch absolute RGB error',tile,tile_h)])
    bottom=cv2.copyMakeBorder(bottom,0,0,0,width-bottom.shape[1],cv2.BORDER_CONSTANT,value=(25,25,25))
    canvas=np.vstack([top,bottom])
    canvas=cv2.copyMakeBorder(canvas,68,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    coverage=state['covered'].sum()/max(int(mask.sum()),1)
    score=float(np.quantile(state['map'][state['covered']],.99))
    for i,text in enumerate([
        f'Patch {state["index"]}/{state["total"]} | box=({x},{y},{w},{h}) | mask covered={coverage:.1%} | partial q99={score:.5f}',
        'Blend: sum(smoothed error x Hann weight) / sum(weights); gray = not covered yet. Playback is not runtime.',
    ]):
        cv2.putText(canvas,text,(10,25+i*27),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--image',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path('reports/r2ad_inference.gif'))
    p.add_argument('--safety-area',choices=['SafetyZone','PLeft','PRight','RoboArm','ConvBelt'],default='SafetyZone')
    p.add_argument('--device',default='auto')
    p.add_argument('--patch-size',type=int,default=224)
    p.add_argument('--stride',type=int,default=64)
    p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--sigma',type=float,default=1.)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--map-vmin',type=float,default=0.)
    p.add_argument('--map-vmax',type=float,default=.5)
    p.add_argument('--fps',type=float,default=3.)
    p.add_argument('--every-n',type=int,default=1,help='Include every Nth patch, plus first and last; all patches are inferred')
    p.add_argument('--width',type=int,default=1024)
    args=p.parse_args()
    if min(args.patch_size,args.stride,args.batch_size,args.every_n)<1 or args.width<720 or args.width%2 or args.fps<=0:
        p.error('Require positive sizes/counts/FPS and an even width >=720')
    if not 0<args.min_coverage<=1 or args.sigma<0 or not 0<=args.map_vmin<args.map_vmax<=1:
        p.error('Invalid mask coverage, sigma or color limits')
    if args.output.suffix.lower()!='.gif':
        p.error('--output must end in .gif')
    cfg=load_config(args.config)
    value=cfg['r2ad']['safety_zone_mask'] if args.safety_area=='SafetyZone' else cfg['data']['mask_types'][args.safety_area]
    mask_path=Path(value).expanduser()
    if not mask_path.is_absolute():
        mask_path=Path(cfg['data']['dataset_base']).expanduser()/mask_path
    frame=cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f'Unreadable image: {args.image}')
    mask=read_mask(mask_path,frame.shape)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    enc,dec,ck=load_models(args.checkpoint,device)
    if ck.get('config',{}).get('safety_area',args.safety_area)!=args.safety_area:
        raise ValueError('Checkpoint safety area differs from requested mask')
    frames=[];visited=[]
    def on_patch(state):
        if state['index']==1 or state['index']%args.every_n==0 or state['index']==state['total']:
            frames.append(render_step(frame,mask,state,visited,args.width,args.map_vmin,args.map_vmax))
            print(f'Animation patch {state["index"]}/{state["total"]}',flush=True)
        visited.append(state['box'])
    result=infer_map(frame,mask,enc,dec,device,args.patch_size,args.stride,args.min_coverage,
                     args.batch_size,args.sigma,on_patch=on_patch)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    durations=[max(10,round(1000/args.fps))]*len(frames);durations[-1]=2000
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    frames[-1].save(args.output.with_suffix('.png'))
    save_map(args.output.with_name(args.output.stem+'_overlay.png'),frame,mask,result,args.map_vmin,args.map_vmax)
    report=dict(image=str(args.image.resolve()),checkpoint=str(args.checkpoint.resolve()),epoch=ck.get('epoch'),
                patches=len(visited),animation_frames=len(frames),score=result['score'],
                covered_mask_fraction=result['coverage_ratio'],settings={
                    k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                note='Same inference/stitching implementation as evaluation. Batches execute together; GIF reveals blending one patch at a time. Partial score can rise or fall. Display colors are GIF-quantized; raw map is saved separately.')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {args.output}; final score={result["score"]:.6f}, coverage={result["coverage_ratio"]:.1%}')


if __name__=='__main__':
    main()
