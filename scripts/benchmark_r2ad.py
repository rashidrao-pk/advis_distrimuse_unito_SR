#!/usr/bin/env python3
"""Benchmark the existing sequential SafetyZone image inference pipeline."""
import argparse
import csv
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.config import load_config
from r2ad.core import images, read_mask, load_models, checkpoint_patch_settings, infer_map


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                min_ms=float(values.min()), max_ms=float(values.max()),
                fps=float(1000 / values.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/cf_dataset_epito.yaml')
    p.add_argument('--input', required=True, help='Image, directory, or image split CSV')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mask', help='Override saved SafetyZone mask')
    p.add_argument('--output-dir', default='results/r2ad/benchmark')
    p.add_argument('--patch-size', type=int)
    p.add_argument('--stride', type=int)
    p.add_argument('--min-coverage', type=float, default=.85)
    p.add_argument('--sigma', type=float, default=1.)
    p.add_argument('--batch-size', type=int, default=32, help='Patches per GPU batch, not frames')
    p.add_argument('--device', default='cuda')
    p.add_argument('--frames', type=int, default=200)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.frames <= 0 or args.warmup < 0 or args.batch_size <= 0:
        p.error('Require frames/batch-size > 0 and warmup >= 0')
    cfg = load_config(args.config)
    entry = args.mask or (cfg.get('r2ad') or {}).get('safety_zone_mask') or cfg['data'].get('mask_types', {}).get('SafetyZone')
    if not entry:
        p.error('No saved SafetyZone mask configured')
    mask_path = Path(entry).expanduser()
    if not mask_path.is_absolute():
        mask_path = Path(cfg['data']['dataset_base']).expanduser() / mask_path
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.set_device(device)
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    sync()
    start = time.perf_counter()
    enc, dec, ck = load_models(args.checkpoint, str(device))
    sync()
    load_seconds = time.perf_counter() - start
    args.patch_size, args.stride = checkpoint_patch_settings(ck, args.patch_size, args.stride)
    paths = images(args.input)
    if not paths:
        p.error('No source frames')
    rng = np.random.default_rng(args.seed)
    selected = [paths[int(i)] for i in rng.permutation(len(paths))[:args.frames]]

    def run(path):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Unreadable frame: {path}')
        mask = read_mask(mask_path, frame.shape)
        sync()
        ready = time.perf_counter()
        result = infer_map(frame, mask, enc, dec, str(device), args.patch_size,
                           args.stride, args.min_coverage, args.batch_size, args.sigma)
        sync()
        done = time.perf_counter()
        return result, ready, done, frame.shape

    for i in range(args.warmup):
        run(selected[i % len(selected)])
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for i, path in enumerate(selected):
        sync()
        start = time.perf_counter()
        result, ready, done, shape = run(path)
        rows.append(dict(file=str(path), width=shape[1], height=shape[0],
                         decode_and_mask_ms=(ready-start)*1000,
                         inference_ms=(done-ready)*1000, frame_ms=(done-start)*1000,
                         patches=len(result['boxes']), coverage=result['coverage_ratio'], score=result['score']))
        del result
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'frame_timings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = dict(settings=vars(args), device=str(device), gpu=torch.cuda.get_device_name(device) if cuda else None,
                   torch_version=torch.__version__, opencv_version=cv2.__version__, platform=platform.platform(),
                   precision='float32; existing infer_map without autocast',
                   checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_epoch=ck.get('epoch'),
                   mask=str(mask_path.resolve()), mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                   parameters=sum(x.numel() for m in (enc, dec) for x in m.parameters()),
                   measured_frames=len(rows), warmup_frames=args.warmup, model_load_seconds=load_seconds,
                   inference=timing_summary([r['inference_ms'] for r in rows]),
                   frame_with_decode_and_mask=timing_summary([r['frame_ms'] for r in rows]),
                   sequential_loop_fps=len(rows)/wall_seconds,
                   mean_cpu_cores_used=cpu_seconds/wall_seconds,
                   process_lifetime_peak_rss_mib=rss/(1024**2 if sys.platform=='darwin' else 1024),
                   min_coverage=min(r['coverage'] for r in rows),
                   patch_count_range=[min(r['patches'] for r in rows), max(r['patches'] for r in rows)],
                   gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if cuda else None,
                   gpu_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if cuda else None,
                   notes=['Sequential frame processing using the unchanged evaluation scorer.',
                          'Inference includes CPU patch extraction, transfers, model, stitching and frame score.',
                          'Frame timing includes image decoding and mask loading, excludes model load, warmup, logging and output writes.',
                          'Repeated reads may use OS filesystem cache. No acquisition, ROS decoding, map rendering, or network publishing.',
                          'GPU memory is PyTorch allocator memory, not total device usage. CPU metric covers this process only.'])
    (out/'benchmark.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print('[saved]', out)


if __name__ == '__main__':
    main()
