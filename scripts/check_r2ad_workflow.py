#!/usr/bin/env python3
"""Read-only checks for R2AD data and run artifacts; no training or file changes."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.frame_manifest import read_frame_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/cf_dataset_epito.yaml')
    parser.add_argument('--split-dir', type=Path, help='Override configured split locations')
    parser.add_argument('--run-dir', type=Path, help='Directory containing model_r2ad.pt and calibration/test outputs')
    parser.add_argument('--safety-area', default='SafetyZone')
    args = parser.parse_args()
    cfg = load_config(args.config)
    rcfg = cfg.get('r2ad', {})
    root = Path(cfg['data']['dataset_base']).expanduser()

    def configured(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    def status(state, step, detail):
        print(f'[{state}] {step}: {detail}', flush=True)

    status('CHECK', 'Configuration', str(Path(args.config).resolve()))
    for key in ('normal_source', 'unexpected_source'):
        source = configured(rcfg[key]) if rcfg.get(key) else None
        present = source is not None and source.is_dir() and any(source.glob('scenario_*/*'))
        status('REVIEW' if present else 'PENDING', key,
               f'{source}; verify every scenario and full extraction logs (file presence cannot prove completeness)')
    mask = rcfg.get('safety_zone_mask') if args.safety_area == 'SafetyZone' else cfg['data']['mask_types'].get(args.safety_area)
    status('PRESENT' if mask and configured(mask).is_file() else 'PENDING', 'Mask',
           f'{mask}; inspect alignment and patch coverage visually')
    records = {}
    for split, key in [('train', 'train_frames'), ('val', 'val_frames'),
                       ('test', 'test_frames'), ('normal_test', 'normal_test_frames')]:
        path = args.split_dir / f'{split}.csv' if args.split_dir else configured(rcfg[key]) if rcfg.get(key) else None
        if path is None or not path.is_file():
            status('PENDING', split, f'Missing CSV: {path}')
            continue
        try:
            rows = read_frame_manifest(path, with_metadata=True)
            labels = Counter(r.get('label', '') for r in rows)
            if not rows:
                raise ValueError('Empty split')
            if any(r.get('split') != ('test' if split == 'normal_test' else split) for r in rows):
                raise ValueError('Rows assigned to wrong split')
            if split in ('train', 'val', 'normal_test') and set(labels) != {'normal'}:
                raise ValueError(f'Expected normal-only rows: {dict(labels)}')
            if split == 'test' and not set(labels).issubset({'normal', 'anomalous'}):
                raise ValueError('Expected annotation-aware normal/anomalous labels')
            records[split] = rows
            status('PASS', split, f'{len(rows)} frames; labels={dict(labels)}; all sources exist')
            if split == 'test' and set(labels) != {'normal', 'anomalous'}:
                status('REVIEW', 'Test classes', 'Both classes needed for AUROC/AUPR; first-50 extraction may miss anomalies')
        except (ValueError, OSError) as exc:
            status('FAIL', split, str(exc))
    if all(s in records for s in ('train', 'val', 'test')):
        groups = {s: {r.get('scenario') or str(r['source'].parent) for r in records[s]}
                  for s in ('train', 'val', 'test')}
        overlaps = {f'{a}/{b}': sorted(groups[a] & groups[b])
                    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
        status('FAIL' if any(overlaps.values()) else 'PASS', 'Scenario separation', str(overlaps))
        normal_sources = {r['source'] for r in records.get('normal_test', [])}
        expected = {r['source'] for r in records['test'] if r['label'] == 'normal'}
        status('PASS' if normal_sources == expected else 'FAIL', 'Normal test subset',
               'Compare normal_test.csv with normal rows of test.csv')
    split_dir = args.split_dir or (configured(rcfg['train_frames']).parent if rcfg.get('train_frames') else None)
    metadata = split_dir / 'metadata.json' if split_dir else None
    if metadata and metadata.is_file():
        data = json.loads(metadata.read_text())
        status('PRESENT' if data.get('annotations') else 'REVIEW', 'Annotation provenance',
               str(data.get('annotations') or 'Split created without --annotations'))
        status('CHECK', 'Split strategy', str(data.get('split_strategy')))
    run = args.run_dir or (Path(cfg.get('results', {}).get('output_dir', 'results')) /
                          str(rcfg.get('dataset_version', root.name)) / 'r2ad' / args.safety_area)
    for step, relative in [('Checkpoint', 'model_r2ad.pt'),
                           ('Monitoring', 'live_monitor/batch_losses.csv'),
                           ('Calibration', 'calibration/threshold.json'),
                           ('Evaluation', 'test/scores.csv'), ('Metrics', 'test/metrics.json'),
                           ('Stress test', 'robustness/robustness.csv')]:
        path = run / relative
        exists = path.is_file() and path.stat().st_size > 0
        status('PRESENT' if exists else 'PENDING', step, str(path))
    status('REVIEW', 'Run completion', 'PRESENT means an artifact exists, not that the run finished or its provenance matches. Check final logs and the runbook gates.')


if __name__ == '__main__':
    main()
