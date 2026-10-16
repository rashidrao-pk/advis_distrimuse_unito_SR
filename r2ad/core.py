"""R2AD shared sampling, model, reconstruction, scoring and stitching.

Masks define regions in the ORIGINAL frame's pixel coordinates. All coordinates
are (x,y,w,h). RGB model input is normalized to [-1,+1].
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IMAGE_EXT={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

def images(root):
    p=Path(root)
    if p.is_file(): return [p]
    if not p.exists(): raise FileNotFoundError(p)
    return sorted(x for x in p.rglob('*') if x.suffix.lower() in IMAGE_EXT)

def read_mask(path, shape):
    mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise FileNotFoundError(f'Mask missing/unreadable: {path}')
    h,w=shape[:2]
    if mask.shape!=(h,w):
        raise ValueError(f'Mask dimensions {mask.shape} do not match frame {(h,w)}. Use an original-frame mask (no silent resize).')
    return (mask>127).astype(np.uint8)

def boxes(mask, patch, stride, min_coverage=.85):
    h,w=mask.shape
    if h<patch or w<patch:
        raise ValueError(f'Frame {w}x{h} smaller than source patch {patch}; choose smaller patch.')
    ys=list(range(0,h-patch+1,stride)); xs=list(range(0,w-patch+1,stride))
    if ys[-1]!=h-patch:ys.append(h-patch)
    if xs[-1]!=w-patch:xs.append(w-patch)
    ii=cv2.integral(mask.astype(np.uint8))
    out=[]
    for y in ys:
        for x in xs:
            covered=int(ii[y+patch,x+patch]-ii[y,x+patch]-ii[y+patch,x]+ii[y,x])
            if covered/(patch*patch)>=min_coverage:out.append((x,y,patch,patch))
    if not out: raise ValueError('No rectangles satisfy mask coverage; lower --min-coverage, choose a smaller --patch-size, or check the mask.')
    return out

def random_boxes(mask, patch, count, rng, min_coverage=.85, scale=(.9,1.1), tries=250):
    h,w=mask.shape; valid=np.argwhere(mask>0)
    if not len(valid): raise ValueError('Mask has no positive pixels')
    out=[]
    for _ in range(count):
        accepted=None
        for _ in range(tries):
            p=min(h,w,max(16,int(round(patch*rng.uniform(*scale)))))
            if p>min(h,w):continue
            cy,cx=valid[int(rng.integers(len(valid)))]
            x=int(np.clip(cx-p//2,0,w-p)); y=int(np.clip(cy-p//2,0,h-p))
            coverage=mask[y:y+p,x:x+p].mean()
            if coverage>=min_coverage:accepted=(x,y,p,p);break
        if accepted is None:
            # deterministic fallback ensures feasible narrow/irregular ROI failures are explicit
            candidates=boxes(mask,min(patch,h,w),max(1,patch//4),min_coverage)
            accepted=candidates[int(rng.integers(len(candidates)))]
        out.append(accepted)
    return out

def extract(frame, box, size=128):
    x,y,w,h=box
    patch=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
    patch=cv2.resize(patch,(size,size),interpolation=cv2.INTER_AREA if w>size else cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(patch.transpose(2,0,1))).float().div(127.5).sub(1.)

class Encoder(nn.Module):
    def __init__(self,z_size=64):
        super().__init__()
        layers=[]; channels=[3,64,128,256,512]
        for a,b in zip(channels[:-1],channels[1:]):layers += [nn.Conv2d(a,b,4,2,1),nn.ReLU()]
        self.conv_layers=nn.Sequential(*layers,nn.Flatten())
        self.fc_mu=nn.Linear(512*8*8,z_size)
        self.fc_logvar=nn.Linear(512*8*8,z_size)
    def forward(self,x):
        h=self.conv_layers(x)
        return self.fc_mu(h),self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self,z_size=64):
        super().__init__()
        self.fc=nn.Linear(z_size,512*8*8)
        self.deconv_layers=nn.Sequential(nn.ReLU(),nn.Unflatten(1,(512,8,8)),
            nn.ConvTranspose2d(512,256,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1),nn.Tanh())
    def forward(self,z):return self.deconv_layers(self.fc(z))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(nn.Conv2d(3,32,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.ReLU(),nn.MaxPool2d(2))
        self.fc_layers=nn.Sequential(nn.Flatten(),nn.Linear(128*16*16,256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,1))
    def forward(self,x):return self.fc_layers(self.conv_layers(x))

def build_models(z,device):return Encoder(z).to(device),Decoder(z).to(device),Discriminator().to(device)

def load_models(checkpoint,device):
    try: ck=torch.load(checkpoint,map_location=device,weights_only=False)
    except TypeError:ck=torch.load(checkpoint,map_location=device)
    z=int(ck.get('latent_dims',ck.get('config',{}).get('latent_dims',64)))
    enc,dec,_=build_models(z,device)
    enc.load_state_dict(ck['encoder_state_dict']);dec.load_state_dict(ck['decoder_state_dict'])
    enc.eval();dec.eval()
    return enc,dec,ck

def transform_frame(frame, tx=0,ty=0,angle=0,scale=1):
    h,w=frame.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,scale)
    M[:,2] += (tx,ty)
    return cv2.warpAffine(frame,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

def infer_map(frame,mask,enc,dec,device,patch=224,stride=64,coverage=.85,batch=32,sigma=1.):
    h,w=frame.shape[:2]
    coords=boxes(mask,patch,stride,coverage)
    accum=np.zeros((h,w),np.float32);weights=np.zeros((h,w),np.float32)
    # Keep window weights strictly positive at boundaries; no divide-by-zero seams.
    hann=np.outer(np.hanning(patch),np.hanning(patch)).astype(np.float32)
    hann=np.maximum(hann,.01)
    patch_scores=[]
    for begin in range(0,len(coords),batch):
        chunk=coords[begin:begin+batch]
        inp=torch.stack([extract(frame,b) for b in chunk]).to(device)
        with torch.inference_mode():
            mu,_=enc(inp); pred=dec(mu)
            residual=(inp-pred).abs().mean(1).cpu().numpy()/2.0
        for b,res in zip(chunk,residual):
            x,y,p,_=b
            resized=cv2.resize(res,(p,p),interpolation=cv2.INTER_LINEAR)
            if sigma>0:resized=cv2.GaussianBlur(resized,(0,0),sigma)
            accum[y:y+p,x:x+p] += resized*hann[:p,:p] if p==patch else resized*np.maximum(np.outer(np.hanning(p),np.hanning(p)),.01)
            weights[y:y+p,x:x+p] += hann[:p,:p] if p==patch else np.maximum(np.outer(np.hanning(p),np.hanning(p)),.01)
            patch_scores.append(float(np.quantile(res,.99)))
    valid=(weights>0)&(mask>0)
    amap=np.divide(accum,weights,out=np.zeros_like(accum),where=weights>0)
    amap[mask==0]=0
    if not valid.any():raise ValueError('No valid pixels after patch stitching')
    score=float(np.quantile(amap[valid],.99))
    return {'score':score,'map':amap,'covered':valid,'boxes':coords,'patch_scores':patch_scores,'coverage_ratio':float(valid.sum()/max(mask.sum(),1))}

def save_map(out,frame,mask,result):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    amap=result['map'];valid=result['covered']
    p99=float(np.quantile(amap[valid],.995)) if valid.any() else 1
    heat=cv2.applyColorMap(np.uint8(np.clip(amap/max(p99,1e-9),0,1)*255),cv2.COLORMAP_JET)
    overlay=frame.copy(); overlay[valid]=cv2.addWeighted(frame,.6,heat,.4,0)[valid]
    cv2.imwrite(str(out),overlay)
    cv2.imwrite(str(out.with_name(out.stem+'_heat.png')),heat)
    np.save(str(out.with_suffix('.npy')),amap)
