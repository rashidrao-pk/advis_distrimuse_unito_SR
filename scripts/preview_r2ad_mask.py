#!/usr/bin/env python3
"""Preview a full-resolution R2AD mask with grid or random training patches."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.core import boxes, random_boxes, read_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--mask', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--min-coverage', type=float, default=.85)
    parser.add_argument('--sampling', choices=['grid', 'random'], default='grid')
    parser.add_argument('--num-patches', type=int, default=48,
                        help='Number of random patches (may overlap), default 48')
    parser.add_argument('--seed', type=int, default=42,
                        help='Reproduce random patch locations and sizes')
    args = parser.parse_args()
    if args.patch_size < 1 or args.stride < 1 or not 0 < args.min_coverage <= 1:
        parser.error('Require positive patch-size/stride and 0 < min-coverage <= 1')
    if args.num_patches < 1:
        parser.error('num-patches must be positive')
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f'Unreadable source image: {args.image}')
    mask = read_mask(args.mask, frame.shape).astype(bool)
    selected = int(mask.sum())
    if not selected:
        raise ValueError('Mask is empty')
    if args.sampling == 'random':
        # Match RectangleDataset: random positions and 0.9--1.1 size scaling.
        rectangles = random_boxes(mask, args.patch_size, args.num_patches,
                                  np.random.default_rng(args.seed), args.min_coverage)
    else:
        rectangles = boxes(mask, args.patch_size, args.stride, args.min_coverage)
    covered = np.zeros(mask.shape, dtype=bool)
    for x, y, w, h in rectangles:
        covered[y:y+h, x:x+w] = True
    overlay = frame.copy()
    overlay[mask] = (.55 * frame[mask] + .45 * np.array([30, 220, 30])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    masked = np.where(mask[..., None], frame, 0)
    sampling = (frame.astype(float) * .3).astype(np.uint8)
    sampling[mask & covered] = (.4 * frame[mask & covered] + .6 * np.array([30, 220, 30])).astype(np.uint8)
    sampling[mask & ~covered] = (0, 0, 255)
    for x, y, w, h in rectangles:
        cv2.rectangle(sampling, (x, y), (x+w-1, y+h-1), (0, 200, 255), 1)
    fraction = selected / mask.size
    coverage = float((mask & covered).sum() / selected)
    info = dict(image=str(args.image.resolve()), mask=str(args.mask.resolve()),
                width=frame.shape[1], height=frame.shape[0], mask_pixels=selected,
                frame_pixels=mask.size, mask_fraction=fraction,
                patch_size=args.patch_size, stride=args.stride, min_coverage=args.min_coverage,
                sampling=args.sampling, seed=args.seed if args.sampling == 'random' else None,
                patch_count=len(rectangles), covered_mask_fraction=coverage,
                rectangles_xywh=rectangles)
    if args.sampling == 'grid':
        info['inference_patches'] = len(rectangles)
    else:
        info['random_size_scale'] = [.9, 1.1]
    sampling_label = 'Grid inference' if args.sampling == 'grid' else f'Random seed={args.seed}'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for name, image, title in [
        ('original', frame, 'Original full-resolution frame'),
        ('overlay', overlay, f'SafetyZone: {fraction:.1%} of frame (green)'),
        ('masked', masked, 'SafetyZone only; outside shown black'),
        ('patch_coverage', sampling, f'{sampling_label}: {len(rectangles)} patches; {coverage:.1%} covered; red = uncovered'),
    ]:
        path = args.output_dir / f'{name}.png'
        if not cv2.imwrite(str(path), image):
            raise IOError(f'Failed writing {path}')
        width = 960
        height = round(image.shape[0] * width / image.shape[1])
        panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(panel, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(panel, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .57, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    contact_sheet = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    if not cv2.imwrite(str(args.output_dir / 'preview.png'), contact_sheet):
        raise IOError('Failed writing contact sheet')
    (args.output_dir / 'coverage.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != 'rectangles_xywh'}, indent=2))
    print(f'Preview: {args.output_dir / "preview.png"}')


if __name__ == '__main__':
    main()
