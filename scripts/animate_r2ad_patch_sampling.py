#!/usr/bin/env python3
"""Model-free source crop, resizing, augmentation and grid-coverage preview."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import boxes,read_mask,extract
from r2ad.dataset import sample_training_patch,add_augmentation_arguments,augmentation_from_args


def render(original,scene,mask,covered,box,source,input_rgb,index,total,mode,augmentation,width):
    x,y,w,h=box
    overview=scene.copy()
    remaining=(mask>0)&~covered;selected=(mask>0)&covered
    overview[remaining]=(.7*overview[remaining]+.3*np.array([0,0,255])).astype(np.uint8)
    overview[selected]=(.7*overview[selected]+.3*np.array([0,255,0])).astype(np.uint8)
    cv2.rectangle(overview,(x,y),(x+w-1,y+h-1),(0,255,255),3)
    def panel(image,title,out_width,height,nearest=False):
        result=cv2.resize(image,(out_width,height),interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        result=cv2.copyMakeBorder(result,32,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(result,title,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        return result
    half=width//2;scene_h=round(original.shape[0]*half/original.shape[1])
    top=np.hstack([panel(original,'Original scene',half,scene_h),
                   panel(overview,'Green=covered; red=remaining; yellow=current',half,scene_h)])
    tile=width//3;height=min(tile,224)
    crop_mask=(mask[y:y+h,x:x+w]*255).astype(np.uint8)
    crop_mask=cv2.cvtColor(crop_mask,cv2.COLOR_GRAY2BGR)
    network=np.ascontiguousarray((input_rgb[...,::-1]*255).clip(0,255).astype(np.uint8))
    bottom=np.hstack([panel(source,f'{w}x{h} source crop (enlarged)',tile,height,True),
                      panel(network,'128x128 resized model input',tile,height,True),
                      panel(crop_mask,'Mask inside current crop',tile,height,True)])
    bottom=cv2.copyMakeBorder(bottom,0,0,0,width-bottom.shape[1],cv2.BORDER_CONSTANT,value=(25,25,25))
    canvas=np.vstack([top,bottom]);canvas=cv2.copyMakeBorder(canvas,78,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    coverage=float(covered[mask>0].mean())
    if mode=='grid':detail=f'Cumulative grid coverage={coverage:.2%}; no neural reconstruction or anomaly scoring'
    else:detail=f'Current patch coverage={coverage:.2%}; zoom={augmentation["zoom"]:.3f} tx={augmentation["tx"]:.1f}px ty={augmentation["ty"]:.1f}px rotation={augmentation["angle"]:.1f}deg'
    for i,line in enumerate([f'{mode.upper()} patch {index}/{total} | box=({x},{y},{w},{h}) | source -> 128x128 input',detail]):
        cv2.putText(canvas,line,(10,27+i*29),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    p.add_argument('--image',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--patch-size',type=int,default=64)
    p.add_argument('--sampling',choices=['random','grid'],default='random')
    p.add_argument('--stride',type=int,default=32)
    p.add_argument('--num-patches',type=int,default=24)
    p.add_argument('--min-coverage',type=float,default=.85)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fps',type=float,default=3.)
    p.add_argument('--every-n',type=int,default=1)
    p.add_argument('--width',type=int,default=1024)
    add_augmentation_arguments(p)
    args=p.parse_args()
    if args.patch_size<16 or min(args.stride,args.num_patches,args.every_n)<1 or args.width<720 or args.width%2 or args.fps<=0 or not 0<args.min_coverage<=1:
        p.error('Require patch >=16, positive counts/stride/FPS, even width >=720 and occupancy in (0,1]')
    if args.output.suffix.lower()!='.gif':p.error('--output must end in .gif')
    settings=augmentation_from_args(args)
    # This tool visualizes a fixed source size, independent of camera zoom.
    settings['patch_scale_min']=settings['patch_scale_max']=1.
    cfg=load_config(args.config)
    value=Path(cfg['r2ad']['safety_zone_mask']).expanduser()
    if not value.is_absolute():value=Path(cfg['data']['dataset_base']).expanduser()/value
    original=cv2.imread(str(args.image))
    if original is None:raise ValueError(f'Unreadable image: {args.image}')
    mask=read_mask(value,original.shape)
    if not mask.any():raise ValueError('Mask is empty')
    if args.patch_size>min(mask.shape):p.error('Patch exceeds frame dimensions')
    rectangles=boxes(mask,args.patch_size,args.stride,args.min_coverage) if args.sampling=='grid' else None
    total=len(rectangles) if rectangles else args.num_patches
    frames=[];records=[];coverage=np.zeros(mask.shape,bool)
    for index in range(total):
        if args.sampling=='grid':
            box=rectangles[index];x,y,w,h=box;coverage[y:y+h,x:x+w]=True
            scene=original;current_mask=mask;augmentation={}
            if index and (index+1)%args.every_n and index+1!=total:continue
            source=scene[y:y+h,x:x+w];tensor=extract(scene,box)
        else:
            tensor,info=sample_training_patch(original,mask,args.patch_size,args.min_coverage,
                np.random.default_rng(args.seed+index),details=True,settings=settings)
            scene=info['scene'];current_mask=info['mask'];box=info['box'];augmentation=info['augmentation']
            source=info['source_patch_bgr'];x,y,w,h=box
            coverage=np.zeros(mask.shape,bool);coverage[y:y+h,x:x+w]=True
            if index and (index+1)%args.every_n and index+1!=total:continue
        input_rgb=((tensor.permute(1,2,0).numpy()+1)/2).clip(0,1)
        frames.append(render(original,scene,current_mask,coverage,box,source,input_rgb,index+1,total,args.sampling,augmentation,args.width))
        records.append(dict(index=index+1,box=list(box),augmentation=augmentation,
                            coverage=float(coverage[current_mask>0].mean())))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    durations=[max(10,round(1000/args.fps))]*len(frames);durations[-1]=2000
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    frames[-1].save(args.output.with_suffix('.png'))
    report=dict(mode=args.sampling,total_patches=total,animation_frames=len(frames),source_patch_size=args.patch_size,
                model_input_size=128,mask_pixels=int(mask.sum()),samples=records,
                actual_augmentation_settings=settings if args.sampling=='random' else None,
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                note='Geometry-only preview: no trained model, reconstruction, anomaly map or optimizer updates. Grid coverage is cumulative; random coverage resets because augmentation changes coordinates each sample. Crops are fixed size; camera zoom remains independent.')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(f'Saved {args.output}; {total} patches, {len(frames)} GIF frames',flush=True)


if __name__=='__main__':main()
