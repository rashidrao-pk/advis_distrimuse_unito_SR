#!/usr/bin/env python3
"""Reflow an existing portrait full-frame GIF without repeating model inference."""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r2ad.frame_preview import append_panels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--split-height',type=int,help='Height of top patch section; auto-detected from source frame in companion JSON when available')
    args=p.parse_args()
    if args.input.resolve()==args.output.resolve():p.error('Choose a different output filename to retain the original GIF')
    if args.output.suffix.lower()!='.gif':p.error('--output must end in .gif')
    metadata_path=args.input.with_suffix('.json')
    metadata=json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata.get('settings',{}).get('layout')=='landscape':p.error('Input metadata already describes a landscape GIF')
    split=args.split_height
    frames=[];durations=[]
    with Image.open(args.input) as animation:
        width,height=animation.size;loop=animation.info.get('loop',0)
        if split is None:
            samples=metadata.get('samples',[])
            source=Path(samples[0]['source']) if samples else None
            if source and source.is_file():
                with Image.open(source) as image:
                    scene_height=round(image.height*(width//2)/image.width)
                split=scene_height+min(width//3,224)+128
            else:p.error('Cannot detect section boundary; supply --split-height (640 for the shown 1024-wide, 1920x1080-source GIF)')
        if not 0<split<height:p.error('--split-height must be between 1 and GIF height-1')
        for index in range(animation.n_frames):
            animation.seek(index)
            frame=animation.convert('RGB')
            base=frame.crop((0,0,width,split));additional=frame.crop((0,split,width,height))
            if width>=720 and split>=68:
                pixels=np.array(base)
                pixels[30:64]=(25,25,25)
                cv2.putText(pixels,'Random patch left; full-grid diagnostic right. Frozen checkpoint, no optimizer updates.',
                            (10,52),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                base=Image.fromarray(pixels)
            frames.append(append_panels(base,additional,'landscape'))
            durations.append(animation.info.get('duration',100))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(args.output,save_all=True,append_images=frames[1:],duration=durations,loop=loop,optimize=False)
    frames[-1].save(args.output.with_suffix('.png'))
    metadata['reformatted_from']=str(args.input.resolve())
    metadata['reformat']=dict(layout='landscape',split_height=split,source_size=[width,height],output_size=list(frames[0].size),frames=len(frames))
    metadata.setdefault('settings',{}).update(layout='landscape',output=str(args.output))
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(f'Saved {args.output}: {width}x{height} -> {frames[0].width}x{frames[0].height}; inference not repeated.')


if __name__=='__main__':main()
