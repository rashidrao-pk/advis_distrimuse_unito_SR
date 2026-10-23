from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from .core import images,read_mask,random_boxes,extract,transform_frame

class RectangleDataset(Dataset):
    def __init__(self,root,mask_path,patch=224,coverage=.85,samples_per_frame=6,augment=True,seed=42,augmentation_mode='crop'):
        self.paths=images(root)
        if not self.paths: raise ValueError(f'No images in {root}')
        self.mask_path=mask_path;self.patch=patch;self.coverage=coverage
        self.samples_per_frame=samples_per_frame;self.augment=augment;self.seed=seed
        if augmentation_mode not in ('crop', 'full'):
            raise ValueError('augmentation_mode must be crop or full')
        self.augmentation_mode=augmentation_mode
        self._mask_cache={}
    def __len__(self):return len(self.paths)*self.samples_per_frame
    def __getitem__(self,index):
        path=self.paths[index//self.samples_per_frame]
        frame=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if frame is None:raise RuntimeError(f'Cannot read {path}')
        shape=frame.shape[:2]
        if shape not in self._mask_cache:
            self._mask_cache[shape]=read_mask(self.mask_path,frame.shape)
        mask=self._mask_cache[shape]
        # Non-deterministic across epochs/workers but reproducible by worker RNG state.
        rng=np.random.default_rng(np.random.randint(0,2**31))
        if self.augment:
            # Apply matched geometric transform to image AND mask, so sampling region stays valid.
            tx=float(rng.uniform(-.03,.03)*frame.shape[1]);ty=float(rng.uniform(-.03,.03)*frame.shape[0]);angle=float(rng.uniform(-2,2))
            # Image and mask alignment must remain identical. Use affine matrix manually for binary mask.
            h,w=frame.shape[:2];M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
            M[:,2]+=(tx,ty)
            mask=cv2.warpAffine(mask,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            alpha=float(rng.uniform(.85,1.15)); beta=float(rng.uniform(-12,12))
            if self.augmentation_mode == 'full':
                frame=cv2.warpAffine(frame,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
                frame=np.uint8(np.clip(frame.astype(np.float32)*alpha+beta,0,255))
        box=random_boxes(mask,self.patch,1,rng,self.coverage)[0]
        if self.augment and self.augmentation_mode == 'crop':
            # Same transformed-frame coordinates and mask, but render only the selected ROI.
            x,y,w,h=box
            local=M.copy();local[:,2]-=(x,y)
            frame=cv2.warpAffine(frame,local,(w,h),borderMode=cv2.BORDER_REFLECT_101)
            frame=np.uint8(np.clip(frame.astype(np.float32)*alpha+beta,0,255))
            box=(0,0,w,h)
        return extract(frame,box)
