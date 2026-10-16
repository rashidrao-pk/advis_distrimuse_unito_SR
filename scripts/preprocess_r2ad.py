#!/usr/bin/env python3
"""Build SafetyZone and lossless normal/test image sets for R2AD.

Usage:
  python scripts/preprocess_r2ad.py --config configs/cf_dataset_epito.yaml --normal-source /path/to/normal/full_frames --unexpected-source /path/to/unexpected/full_frames

Default split strategy is grouped (top-level scenario folders). It fails if
insufficient independent groups exist rather than leaking near-identical video
frames across training/validation/test. --split-mode temporal explicitly enables
contiguous frame blocks, which are useful for debugging but weaker for evaluation.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config, DEFAULT_CONFIG
from r2ad.core import images
from r2ad.safetyzone import create_safety_zone


def split_normal(paths, source, val_fraction, test_fraction, seed, mode):
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError('Require 0 < validation/test fractions and total < 1')
    if mode == 'grouped':
        groups = defaultdict(list)
        source = Path(source)
        for p in paths:
            relative = p.relative_to(source)
            # First subdirectory is treated as an independent recording/scenario.
            if len(relative.parts) < 2:
                raise ValueError('Grouped mode requires source/scenario_name/frame.jpg. Use separate scenario directories, or explicitly choose --split-mode temporal.')
            groups[relative.parts[0]].append(p)
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError(f'Grouped split needs at least 3 independent scenario folders, found {len(keys)}. Use --split-mode temporal only for a preliminary smoke test.')
        random.Random(seed).shuffle(keys)
        n_test = max(1, round(len(keys) * test_fraction))
        n_val = max(1, round(len(keys) * val_fraction))
        if n_test + n_val >= len(keys): raise ValueError('Not enough groups for nonempty train/val/test')
        names = dict(test=keys[:n_test], val=keys[n_test:n_test+n_val], train=keys[n_test+n_val:])
        return {k: sorted(p for g in v for p in groups[g]) for k, v in names.items()}, names
    if mode == 'temporal':
        ordered = sorted(paths)
        n = len(ordered)
        if n < 3: raise ValueError('Need at least 3 normal frames to split')
        t = max(1, round(n * test_fraction)); v = max(1, round(n * val_fraction))
        if t + v >= n: raise ValueError('Not enough frames for train split')
        return {'train': ordered[:n-t-v], 'val': ordered[n-t-v:n-t], 'test': ordered[n-t:]}, {'warning': 'Temporal blocks from one source may still have correlated train/test content; use independent scenarios for publication.'}
    raise ValueError(mode)


def transfer(paths, source_root, target, mode, manifest, label, split, mask_hw):
    target.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(paths):
        # Names are unique even when multiple scenarios contain frame_0001.png.
        rel = str(src.relative_to(source_root))
        digest = hashlib.sha1(rel.encode()).hexdigest()[:12]
        destination = target / f'{idx:07d}_{digest}{src.suffix.lower()}'
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Output already exists: {destination}; use an empty --output-dir')
        if mode == 'symlink': destination.symlink_to(src.resolve())
        else: shutil.copy2(src, destination)
        manifest.append(dict(split=split, label=label, source=str(src), processed=str(destination)))
    return len(paths)


def verify_shape(paths, shape_hw, description):
    for p in paths:
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None: raise RuntimeError(f'Unreadable {description} frame: {p}')
        if frame.shape[:2] != shape_hw:
            raise ValueError(f'Original frame resolution mismatch {p}: {frame.shape[:2]} versus mask {shape_hw}. Do not resize original camera data automatically.')


def main():
    p = argparse.ArgumentParser(description='Prepare full-resolution R2AD SafetyZone dataset')
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--normal-source', type=Path, help='Normal full-resolution images organized by recording/scenario')
    p.add_argument('--unexpected-source', type=Path, help='Optional unexpected full-resolution frames (test only)')
    p.add_argument('--output-dir', type=Path, help='Defaults to <data.dataset_base>/r2ad/full_frames')
    p.add_argument('--mask-output', type=Path, help='Defaults to dataset_base/masks/Mask_Generation_v6_SafetyZone.png')
    p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--test-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-mode', choices=['grouped', 'temporal'], default='grouped')
    p.add_argument('--transfer', choices=['symlink', 'copy'], default='symlink')
    p.add_argument('--mask-only', action='store_true', help='Build union mask without preparing datasets')
    args = p.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['data']['dataset_base']).expanduser()
    extra = cfg.get('r2ad', {})
    mask_out = args.mask_output or extra.get('safety_zone_mask')
    info = create_safety_zone(args.config, output=mask_out)
    print('[SafetyZone]', json.dumps(info, indent=2))
    if args.mask_only: return
    normal_source = args.normal_source or extra.get('normal_source')
    if not normal_source: p.error('Set --normal-source or r2ad.normal_source (full-resolution normal frames)')
    normal_source = Path(normal_source).expanduser().resolve()
    abnormal_source = args.unexpected_source or extra.get('unexpected_source')
    abnormal_source = Path(abnormal_source).expanduser().resolve() if abnormal_source else None
    output = args.output_dir or extra.get('preprocessed_dir') or (root / 'r2ad' / 'full_frames')
    output = Path(output).expanduser().resolve()
    if output == normal_source or normal_source in output.parents:
        raise ValueError('Output must not be nested under normal source directory')
    if abnormal_source and (output == abnormal_source or abnormal_source in output.parents):
        raise ValueError('Output must not be nested under unexpected source directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to mix datasets in nonempty output directory: {output}')
    normals = images(normal_source)
    if not normals: raise ValueError(f'No normal images under {normal_source}')
    unexpected = images(abnormal_source) if abnormal_source else []
    # Verify every input, not only a sample, before writing file links.
    shape_hw = (info['height'], info['width'])
    verify_shape(normals, shape_hw, 'normal')
    verify_shape(unexpected, shape_hw, 'unexpected')
    splits, provenance = split_normal(normals, normal_source, args.val_fraction, args.test_fraction, args.seed, args.split_mode)
    manifest = []
    counts = {}
    for split, files in splits.items():
        counts[split] = transfer(files, normal_source, output / split / 'normal', args.transfer, manifest, 'normal', split, shape_hw)
    if unexpected:
        counts['unexpected'] = transfer(unexpected, abnormal_source, output / 'test' / 'unexpected', args.transfer, manifest, 'unexpected', 'test', shape_hw)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['split', 'label', 'source', 'processed'])
        writer.writeheader(); writer.writerows(manifest)
    metadata = dict(safety_zone=info, split_strategy=args.split_mode, provenance=provenance, counts=counts,
                    transfer=args.transfer, seed=args.seed, normal_source=str(normal_source),
                    unexpected_source=str(abnormal_source) if abnormal_source else None)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('[data]', json.dumps(counts, indent=2))
    print('[output]', output)
    print('Set r2ad.train_frames / val_frames / test_frames / normal_test_frames to:')
    print('  train_frames: ' + str(output / 'train' / 'normal'))
    print('  val_frames: ' + str(output / 'val' / 'normal'))
    print('  test_frames: ' + str(output / 'test'))
    print('  normal_test_frames: ' + str(output / 'test' / 'normal'))
    print('  safety_zone_mask: ' + info['output'])

if __name__ == '__main__': main()
