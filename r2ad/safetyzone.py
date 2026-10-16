"""Union original frame-resolution binary safety masks; preserve source images."""
from pathlib import Path
import cv2
import numpy as np
from .config import load_config, DEFAULT_CONFIG

SOURCE_AREAS = ('PLeft', 'PRight', 'RoboArm', 'ConvBelt')

def union_masks(mask_paths, threshold=127):
    combined = None
    binary = {}
    for name, path in mask_paths.items():
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f'Cannot read mask for {name}: {path}')
        img = (raw > threshold).astype(np.uint8)
        if combined is not None and img.shape != combined.shape:
            raise ValueError(f'Inconsistent mask dimensions: {name} {img.shape} vs {combined.shape}; refusing to silently resize masks')
        combined = img.copy() if combined is None else np.maximum(combined, img)
        binary[name] = img
    if combined is None or not np.any(combined):
        raise ValueError('Combined SafetyZone mask is empty')
    return combined, binary

def create_safety_zone(config_path=DEFAULT_CONFIG, output=None):
    cfg = load_config(config_path)
    root = Path(cfg['data']['dataset_base']).expanduser()
    mask_cfg = cfg['data']['mask_types']
    paths = {}
    for name in SOURCE_AREAS:
        if name not in mask_cfg:
            raise KeyError(f'Missing data.mask_types.{name}')
        p = Path(mask_cfg[name]).expanduser()
        paths[name] = p if p.is_absolute() else root / p
    combined, component_masks = union_masks(paths)
    output = Path(output) if output else root / 'masks' / 'Mask_Generation_v6_SafetyZone.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), combined * 255):
        raise IOError(f'Unable to save mask at {output}')
    # Greyscale preview with boundary contours. Source images are never modified.
    overlay = np.zeros((*combined.shape, 3), dtype=np.uint8)
    overlay[combined > 0] = (210, 210, 210)
    palette = [(255, 0, 0), (0, 170, 255), (0, 255, 0), (255, 0, 255)]
    for (name, mask), color in zip(component_masks.items(), palette):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
    preview = output.with_name(output.stem + '_preview.png')
    cv2.imwrite(str(preview), overlay)
    info = dict(output=str(output), preview=str(preview), width=int(combined.shape[1]),
                height=int(combined.shape[0]), zone_fraction=float(combined.mean()),
                area_fractions={k: float(m.mean()) for k, m in component_masks.items()})
    return info
