"""Full-scene diagnostic panels using one stitched map and aligned area masks."""
import cv2
import numpy as np
from matplotlib import colormaps
from PIL import Image

from .core import infer_map

AREAS=('PLeft','PRight','RoboArm','ConvBelt')


def full_frame_preview(scene,combined_mask,area_masks,enc,dec,device,patch,stride,coverage,batch,sigma,
                       width,vmin,vmax,scope='safety',outside_dim=.35):
    if scope!='safety':raise ValueError('Full-frame inference is restricted to SafetyZone; use scope=safety')
    if not 0<=outside_dim<=1:raise ValueError('outside_dim must be between 0 and 1')
    result=infer_map(scene,combined_mask,enc,dec,device,patch,stride,coverage,batch,sigma,return_reconstruction=True)
    heat=colormaps['Reds'](np.clip((result['map']-vmin)/(vmax-vmin),0,1),bytes=True)[...,:3]
    heat=np.ascontiguousarray(heat[...,::-1])
    def masked_heat(mask):
        image=heat.copy();image[~result['covered']]=(140,140,140);image[(mask==0)|(combined_mask==0)]=(255,255,255)
        return image
    def metrics(mask):
        valid=result['covered']&(mask>0);count=int((mask>0).sum())
        return dict(mask_pixels=count,covered_pixels=int(valid.sum()),
                    coverage=float(valid.sum()/count) if count else 0.,
                    q99=float(np.quantile(result['map'][valid],.99)) if valid.any() else None,
                    mean_error=float(result['map'][valid].mean()) if valid.any() else None)
    scores=dict(full_image=metrics(np.ones_like(combined_mask)),SafetyZone=metrics(combined_mask),
                **{name:metrics(mask) for name,mask in area_masks.items()})
    reconstruction=(scene.astype(np.float32)*outside_dim).astype(np.uint8)
    valid=result['covered']&(combined_mask>0)
    reconstruction[valid]=result['reconstruction'][valid]
    reconstruction[(combined_mask>0)&~result['covered']]=(140,140,140)
    overlay=scene.copy();overlay[combined_mask==0]=reconstruction[combined_mask==0]
    overlay[valid]=cv2.addWeighted(scene,.6,heat,.4,0)[valid]
    contours,_=cv2.findContours(combined_mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(0,255,255),2)
    def panel(image,title,w):
        height=round(scene.shape[0]*w/scene.shape[1])
        tile=cv2.resize(image,(w,height),interpolation=cv2.INTER_AREA)
        tile=cv2.copyMakeBorder(tile,36,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
        cv2.putText(tile,title,(7,24),cv2.FONT_HERSHEY_SIMPLEX,.43 if w>=360 else .35,(255,255,255),1,cv2.LINE_AA)
        return tile
    half=width//2
    first=np.hstack([panel(reconstruction,'ROI reconstruction; outside = faded input',half),
                     panel(masked_heat(combined_mask),f'ROI residual [{vmin:g},{vmax:g}]; white = outside',half)])
    q99=scores['SafetyZone']['q99']
    combined_title=f'Combined SafetyZone q99={q99:.4f}' if q99 is not None else 'Combined SafetyZone: no covered pixels'
    second=np.hstack([panel(overlay,'Combined SafetyZone residual overlay',half),
                      panel(masked_heat(combined_mask),combined_title,half)])
    tiles=[]
    for name in AREAS:
        metric=scores[name];score=f'{metric["q99"]:.3f}' if metric['q99'] is not None else 'N/A'
        tiles.append(panel(masked_heat(area_masks[name]),f'{name} q99={score} cov={metric["coverage"]:.0%}',width//4))
    third=np.hstack(tiles)
    third=cv2.copyMakeBorder(third,0,0,0,width-third.shape[1],cv2.BORDER_CONSTANT,value=(25,25,25))
    panels=np.vstack([first,second,third])
    footer=np.full((54,width,3),25,np.uint8)
    text=['SafetyZone inference only. Outside = faded input / white map; gray = uncovered ROI.',
          'Area panels are views of one model/map; q99 values are not calibrated alarm decisions.']
    for i,line in enumerate(text):
        cv2.putText(footer,line,(8,21+i*24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    panels=np.vstack([panels,footer])
    return Image.fromarray(cv2.cvtColor(panels,cv2.COLOR_BGR2RGB)),dict(
        scope=scope,outside_dim=outside_dim,outside_map_color='white',grid_patch_count=len(result['boxes']),patch_size=patch,stride=stride,sigma=sigma,
        scores=scores,score_definition='q99 of this grid map restricted to each area; not separate detectors or calibrated decisions')


def append_panels(base,additional,layout='portrait'):
    if layout=='landscape':
        canvas=Image.new('RGB',(base.width+additional.width,max(base.height,additional.height)),(25,25,25))
        canvas.paste(base,(0,0));canvas.paste(additional,(base.width,0))
    elif layout=='portrait':
        canvas=Image.new('RGB',(base.width,base.height+additional.height),(25,25,25))
        canvas.paste(base,(0,0));canvas.paste(additional,(0,base.height))
    else:
        raise ValueError('layout must be landscape or portrait')
    return canvas
