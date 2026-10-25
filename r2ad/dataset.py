from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from .core import images,read_mask,random_boxes,extract,transform_frame

DEFAULT_AUGMENTATION=dict(aug_translate=.03,aug_rotate=2.,aug_contrast=.15,aug_brightness=12.,patch_scale_min=.9,patch_scale_max=1.1)


def add_augmentation_arguments(parser):
    for flag,help_text in [
        ('aug-translate','Maximum translation as fraction of full image width/height; 0 disables'),
        ('aug-rotate','Maximum absolute rotation in degrees; 0 disables'),
        ('aug-contrast','Contrast multiplier jitter around 1; 0 disables'),
        ('aug-brightness','Maximum absolute pixel brightness offset; 0 disables'),
        ('patch-scale-min','Minimum random crop size / patch-size (apparent zoom)'),
        ('patch-scale-max','Maximum random crop size / patch-size (apparent zoom)')]:
        parser.add_argument('--'+flag,type=float,default=DEFAULT_AUGMENTATION[flag.replace('-','_')],help=help_text)


def augmentation_from_args(args):
    return validate_augmentation({key:getattr(args,key,default) for key,default in DEFAULT_AUGMENTATION.items()})


def validate_augmentation(settings=None):
    settings=dict(DEFAULT_AUGMENTATION,**(settings or {}))
    if not all(np.isfinite(v) for v in settings.values()):
        raise ValueError('Augmentation ranges must be finite')
    if not 0<=settings['aug_translate']<=1 or not 0<=settings['aug_rotate']<=180 or not 0<=settings['aug_contrast']<=1 or not 0<=settings['aug_brightness']<=255:
        raise ValueError('Require translation/contrast in [0,1], rotation in [0,180], brightness in [0,255]')
    if not 0<settings['patch_scale_min']<=settings['patch_scale_max']:
        raise ValueError('Require 0 < patch-scale-min <= patch-scale-max')
    return settings


def sample_training_patch(frame,mask,patch,coverage,rng,augment=True,augmentation_mode='crop',details=False,settings=None):
    """Shared training sampler; optional geometry for explanatory animations."""
    original=frame
    settings=validate_augmentation(settings)
    if augment:
        translation=settings['aug_translate'];rotation=settings['aug_rotate']
        tx=float(rng.uniform(-translation,translation)*frame.shape[1]);ty=float(rng.uniform(-translation,translation)*frame.shape[0]);angle=float(rng.uniform(-rotation,rotation))
        h,w=frame.shape[:2];M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
        M[:,2]+=(tx,ty)
        mask=cv2.warpAffine(mask,M,(w,h),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        contrast=settings['aug_contrast'];brightness=settings['aug_brightness']
        alpha=float(rng.uniform(1-contrast,1+contrast));beta=float(rng.uniform(-brightness,brightness))
        if augmentation_mode=='full':
            frame=cv2.warpAffine(frame,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
            frame=np.uint8(np.clip(frame.astype(np.float32)*alpha+beta,0,255))
    box=random_boxes(mask,patch,1,rng,coverage,scale=(settings['patch_scale_min'],settings['patch_scale_max']))[0]
    extraction_box=box
    if augment and augmentation_mode=='crop':
        x,y,w,h=box
        local=M.copy();local[:,2]-=(x,y)
        frame=cv2.warpAffine(frame,local,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        frame=np.uint8(np.clip(frame.astype(np.float32)*alpha+beta,0,255))
        extraction_box=(0,0,w,h)
    tensor=extract(frame,extraction_box)
    if not details:
        return tensor
    scene=original
    if augment:
        scene=cv2.warpAffine(original,M,(original.shape[1],original.shape[0]),borderMode=cv2.BORDER_REFLECT_101)
        scene=np.uint8(np.clip(scene.astype(np.float32)*alpha+beta,0,255))
    return tensor,dict(scene=scene,mask=mask,box=box,
                       augmentation=dict(tx=tx,ty=ty,angle=angle,alpha=alpha,beta=beta) if augment else {})


class RectangleDataset(Dataset):
    def __init__(self,root,mask_path,patch=224,coverage=.85,samples_per_frame=6,augment=True,seed=42,augmentation_mode='crop',augmentation_settings=None):
        self.paths=images(root)
        if not self.paths: raise ValueError(f'No images in {root}')
        self.mask_path=mask_path;self.patch=patch;self.coverage=coverage
        self.samples_per_frame=samples_per_frame;self.augment=augment;self.seed=seed
        if augmentation_mode not in ('crop', 'full'):
            raise ValueError('augmentation_mode must be crop or full')
        self.augmentation_mode=augmentation_mode
        self.augmentation_settings=validate_augmentation(augmentation_settings)
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
        return sample_training_patch(frame,mask,self.patch,self.coverage,rng,self.augment,self.augmentation_mode,settings=self.augmentation_settings)
