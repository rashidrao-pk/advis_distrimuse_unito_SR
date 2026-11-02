"""Save representative input geometry before optimization, without consuming RNG state."""
import hashlib
import html
import json
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np

from .core import read_mask
from .dataset import sample_training_patch,DEFAULT_AUGMENTATION


def save_training_preview(train,val,args,directory,resuming=False):
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory=directory/f'{"resume" if resuming else "preview"}_{stamp}'
    directory.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    report=dict(stage='preview_only_no_training_performed' if getattr(args,'preview_only',False) else 'resume_before_next_optimizer_update' if resuming else 'before_first_optimizer_update',
                safety_area=args.safety_area,source_patch_size=args.patch_size,model_input_size=128,
                fixed_patch_size=args.fixed_patch_size,min_patch_occupancy=args.min_coverage,
                samples_per_frame=args.samples_per_frame,mask_source=str(Path(args.mask).resolve()),
                mask_sha256=hashlib.sha256(Path(args.mask).read_bytes()).hexdigest(),config=config,samples=[],
                sampling='random patches; no training grid stride',datasets={
                    name:dict(source_frames=len(dataset.paths) if hasattr(dataset,'paths') else len(dataset.rows),
                              patches_per_epoch=len(dataset),augmented=dataset.augment)
                    for name,dataset in [('train',train),('val',val)]})
    def write(name,image):
        if not cv2.imwrite(str(directory/name),image):raise IOError(f'Cannot write preview {name}')
    def panel(image,title,width,height,nearest=False):
        factor=min(width/image.shape[1],height/image.shape[0])
        resized=cv2.resize(image,(max(1,round(image.shape[1]*factor)),max(1,round(image.shape[0]*factor))),
                           interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        canvas=np.full((height,width,3),25,np.uint8)
        y=(height-resized.shape[0])//2;x=(width-resized.shape[1])//2
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        canvas=cv2.copyMakeBorder(canvas,32,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(canvas,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        return canvas
    overview_saved=False;links=[]
    for split,dataset in [('train',train),('val',val)]:
        rng=np.random.default_rng(args.seed+(0 if split=='train' else 10000))
        count=len(dataset.paths) if hasattr(dataset,'paths') else len(dataset.rows)
        choices=rng.choice(count,args.preview_samples,replace=count<args.preview_samples)
        tiles=[]
        for index,choice in enumerate(choices,1):
            if hasattr(dataset,'paths'):
                source=dataset.paths[int(choice)];frame=cv2.imread(str(source))
                if frame is None:raise ValueError(f'Unreadable preview source: {source}')
            else:
                record=dataset.rows[int(choice)];frame=dataset.reader.read(record)
                source=f'{record["bag"]}:{record["topic"]}:{record["ordinal"]}'
            mask=read_mask(args.mask,frame.shape)
            settings=getattr(dataset,'augmentation_settings',DEFAULT_AUGMENTATION)
            mode=getattr(dataset,'augmentation_mode','full')
            tensor,info=sample_training_patch(frame,mask,dataset.patch,dataset.coverage,rng,dataset.augment,mode,details=True,settings=settings)
            x,y,w,h=info['box'];prefix=f'{split}_{index:02d}'
            original=frame.copy();original[mask>0]=(.7*original[mask>0]+.3*np.array([0,255,0])).astype(np.uint8)
            corners=np.array([[[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]]],np.float64)
            source_corners=cv2.transform(corners,cv2.invertAffineTransform(info['affine_matrix']))[0]
            cv2.polylines(original,[np.rint(source_corners).astype(np.int32)],True,(0,255,255),3)
            augmented=info['scene'].copy();selected=info['mask']>0
            augmented[selected]=(.7*augmented[selected]+.3*np.array([0,255,0])).astype(np.uint8)
            cv2.rectangle(augmented,(x,y),(x+w-1,y+h-1),(0,255,255),3)
            network=np.ascontiguousarray(((tensor.permute(1,2,0).numpy()+1)/2*255).clip(0,255).astype(np.uint8)[...,::-1])
            write(prefix+'_source_crop.png',info['source_patch_bgr'])
            write(prefix+'_model_input.png',network)
            np.save(directory/(prefix+'_model_input.npy'),tensor.numpy())
            write(prefix+'_locations.jpg',np.hstack([panel(original,'Source ROI + mapped crop',960,540),
                                                   panel(augmented,'Augmented ROI + sampled crop',960,540)]))
            tile=np.vstack([np.hstack([panel(original,'Source ROI + mapped crop',480,270),panel(augmented,'Augmented ROI + crop',480,270)]),
                            np.hstack([panel(info['source_patch_bgr'],f'Native {w}x{h} crop (enlarged)',480,224,True),
                                       panel(network,'Actual 128x128 model input',480,224,True)])])
            header=np.full((68,tile.shape[1],3),25,np.uint8)
            title=[f'{split.upper()} {index} | {args.safety_area} | box=({x},{y},{w},{h}) | model=128x128',
                   f'Source: {Path(str(source)).parent.name}/{Path(str(source)).name}']
            for line,text in enumerate(title):cv2.putText(header,text,(8,25+line*27),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
            tile=np.vstack([header,tile]);write(prefix+'_preview.jpg',tile);tiles.append(tile)
            occupancy=float(info['mask'][y:y+h,x:x+w].mean())
            report['samples'].append(dict(split=split,source=str(source),frame_index=int(choice),
                frame_shape_hwc=list(frame.shape),box_xywh=[x,y,w,h],source_crop_shape=list(info['source_patch_bgr'].shape),
                model_input_shape=list(network.shape),source_corners_xy=source_corners.tolist(),
                affine_matrix=info['affine_matrix'].tolist(),augmentation=info['augmentation'],
                augmentation_settings=dict(settings),patch_mask_occupancy=occupancy,
                source_mask_fraction=float(mask.mean()),preview=prefix+'_preview.jpg'))
            if not overview_saved:
                write('mask.png',mask*255);write('area_overlay.jpg',original)
                report.update(frame_resolution_wh=[frame.shape[1],frame.shape[0]],mask_pixels=int(mask.sum()),mask_fraction=float(mask.mean()))
                overview_saved=True
        while len(tiles)%2:tiles.append(np.full_like(tiles[0],25))
        sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
        write(split+'_samples.jpg',sheet)
        if hasattr(dataset,'reader'):dataset.reader.close()
        links.append(f'<h2>{split}</h2><a href="{split}_samples.jpg"><img style="max-width:100%" src="{split}_samples.jpg"></a>')
    report['note']='Representative seeded samples, not a replay of queued optimizer batches. RNG state is unchanged. Coordinates refer to augmented scenes; source quadrilaterals map back through the inverse affine. Native crops and network inputs are saved separately.'
    (directory/'plan.json').write_text(json.dumps(report,indent=2))
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Training input preview</title><h1>Training area and patch setup</h1><pre>'
        +html.escape(f'Area: {args.safety_area}\nSource patch: {args.patch_size}x{args.patch_size}\nModel input: 128x128\nFixed source size: {args.fixed_patch_size}\nMask: {args.mask}')
        +'</pre><p>'+html.escape(report['note'])+'</p>'+''.join(links))
    return directory,report
