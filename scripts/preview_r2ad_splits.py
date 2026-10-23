#!/usr/bin/env python3
"""Preview annotated virtual splits before training; never modify source images."""
import argparse
import html
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.frame_manifest import read_frame_manifest

AREAS = ('PLeft', 'PRight', 'RoboArm', 'ConvBelt')


def scenario(row):
    return row.get('scenario') or row['source'].parent.name.removeprefix('scenario_')


def sample_rows(rows, count, seed):
    """Spread samples across recordings instead of showing adjacent first frames."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[scenario(row)].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while keys and len(selected) < count:
        for key in list(keys):
            selected.append(groups[key].pop())
            if not groups[key]:
                keys.remove(key)
            if len(selected) == count:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cf_dataset_epito.yaml')
    parser.add_argument('--split-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/r2ad_split_preview'))
    parser.add_argument('--samples', type=int, default=6, help='Samples per requested split/label group')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    cfg = load_config(args.config)
    mask_path = Path(cfg['r2ad']['safety_zone_mask']).expanduser()
    if not mask_path.is_absolute():
        mask_path = Path(cfg['data']['dataset_base']).expanduser() / mask_path
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Unreadable SafetyZone mask: {mask_path}')
    mask = mask > 127
    rows = {s: read_frame_manifest(args.split_dir / f'{s}.csv', with_metadata=True)
            for s in ('train', 'val', 'test')}
    issues = []
    for split, records in rows.items():
        allowed = {'normal'} if split != 'test' else {'normal', 'anomalous'}
        if not records or any(r.get('label') not in allowed or r.get('split') != split for r in records):
            issues.append(f'{split}: empty split, incorrect assignment, or unexpected labels')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        shared_files = {r['source'] for r in rows[a]} & {r['source'] for r in rows[b]}
        shared_groups = {scenario(r) for r in rows[a]} & {scenario(r) for r in rows[b]}
        if shared_files or shared_groups:
            issues.append(f'{a}/{b}: {len(shared_files)} shared files; shared scenarios={sorted(shared_groups)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'seed': args.seed, 'issues': issues, 'split_counts': {
        s: dict(Counter(r['label'] for r in records)) for s, records in rows.items()}, 'samples': {}}
    links = []
    for split, label in [('train', 'normal'), ('val', 'normal'), ('test', 'anomalous')]:
        name = f'{split}_{label}'
        selected = sample_rows([r for r in rows[split] if r['label'] == label], args.samples, args.seed)
        if not selected:
            issues.append(f'No {label} frames in {split}; check extraction limits and annotations')
            continue
        panels = []
        report['samples'][name] = []
        for index, row in enumerate(selected):
            image = cv2.imread(str(row['source']))
            if image is None or image.shape[:2] != mask.shape:
                raise ValueError(f'Unreadable frame or mask dimension mismatch: {row["source"]}')
            overlay = image.copy()
            overlay[mask] = (.65 * image[mask] + .35 * np.array([20, 220, 20])).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            width, height = 640, round(image.shape[0] * 640 / image.shape[1])
            panel = np.hstack([cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
                               for im in (image, overlay)])
            panel = cv2.copyMakeBorder(panel, 90, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            text = [f'{split} | {label} | scenario={scenario(row)} | camera={row.get("camera", "?")} | frame={row.get("frame_id", row["source"].stem)}',
                    ' | '.join(f'{area}={row.get("label_" + area, "unknown")}' for area in AREAS),
                    'Original frame (left) | SafetyZone overlay (right); area labels are annotations, not predictions']
            for line, value in enumerate(text):
                cv2.putText(panel, value, (12, 24 + line * 27), cv2.FONT_HERSHEY_SIMPLEX,
                            .53, (255, 255, 255), 1, cv2.LINE_AA)
            filename = f'{name}_{index + 1:02d}.jpg'
            if not cv2.imwrite(str(args.output_dir / filename), panel):
                raise IOError(f'Failed writing {filename}')
            report['samples'][name].append({k: str(v) if isinstance(v, Path) else v for k, v in row.items()})
            panels.append(panel)
        sheet = f'{name}.jpg'
        if not cv2.imwrite(str(args.output_dir / sheet), np.vstack(panels)):
            raise IOError(f'Failed writing {sheet}')
        links.append(f'<h2>{html.escape(name)}</h2><a href="{sheet}"><img style="max-width:100%" src="{sheet}"></a>')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2))
    message = 'Split checks passed; visual inspection still required.' if not issues else '\n'.join(issues)
    (args.output_dir / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R2AD split preview</title>'
        '<h1>R2AD split preview</h1><pre>' + html.escape(message) + '</pre>' + ''.join(links))
    print(message)
    print(f'Preview: {args.output_dir / "index.html"}')
    print(json.dumps(report['split_counts'], indent=2))
    if issues:
        sys.exit(1)


if __name__ == '__main__':
    main()
