#!/usr/bin/env python3
"""Build an MCAP frame manifest, without extracting or duplicating frames.

Within every normal scenario, four sequential frames go to training and the
fifth to validation (ordinal % 5 == 4). Anomalous scenarios go to test.
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ad.mcap_data import FIELDS


def category_for(sid, cfg):
    major = int(sid.split('_')[0])
    for key, value in (cfg.get('scenario_categories') or {}).items():
        m = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?',str(key))
        if m and int(m.group(1)) <= major <= int(m.group(2) or m.group(1)):
            label = str(value).lower()
            if label in ('normal','expected'): return 'normal'
            if label in ('anomalous','unexpected','abnormal','anomaly'): return 'unexpected'
            raise ValueError(f'Unknown category {value!r} for {sid}')
    raise ValueError(f'No category mapping for scenario {sid}')


def index_bag(path, sid, label, topic, stride=1, max_frames=0):
    from mcap.reader import make_reader
    seen = 0
    tie_counts = defaultdict(int)
    rows = []
    with path.open('rb') as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()
        if summary is None or not summary.chunk_indexes:
            print(f'[warning] {path} has no chunk index; time-range seeks may be slow.')
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            if not schema.name.replace('/msg/','/').endswith(('/Image','/CompressedImage')):
                raise ValueError(f'Unexpected topic type: {topic}: {schema.name}')
            stamp = int(message.log_time)
            occurrence = tie_counts[stamp]
            tie_counts[stamp] += 1
            if seen % stride == 0:
                index = len(rows)
                split = 'test' if label == 'unexpected' else ('val' if index % 5 == 4 else 'train')
                rows.append(dict(scenario=sid,label=label,split=split,bag=str(path.resolve()),
                                 topic=topic,log_time=stamp,occurrence=occurrence,ordinal=seen))
                if max_frames and len(rows) >= max_frames:
                    break
            seen += 1
    if not rows:
        raise ValueError(f'No image messages on {topic} in {path}')
    return rows


def main():
    p=argparse.ArgumentParser(description='Create MCAP-native 4:1 per-scenario R2AD manifest')
    p.add_argument('--config',default='configs/cf_dataset_epito.yaml')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--scenario',nargs='+'); group.add_argument('--all',action='store_true')
    p.add_argument('--camera',default='back_view')
    p.add_argument('--output')
    p.add_argument('--save-every-n',type=int,default=1)
    p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    if args.save_every_n<1 or args.max_frames<0:p.error('Invalid frame sampling values')
    cfg=yaml.safe_load(Path(args.config).read_text())
    root=Path(cfg['data']['dataset_base']).expanduser()
    opts=cfg['scenario_options']
    scenarios=list(opts) if args.all else (args.scenario or [str(cfg['scenario']['id'])])
    unknown=set(scenarios)-set(opts)
    if unknown:p.error(f'Unknown scenario IDs: {sorted(unknown)}')
    topic=args.camera if args.camera.startswith('/') else f'/camera/{args.camera}/image_raw'
    output=Path(args.output or (cfg.get('r2ad') or {}).get('manifest') or root/'r2ad'/'manifest.csv')
    if not output.is_absolute():output=root/output
    output.parent.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for sid in scenarios:
        bag=Path(opts[sid]['rosbag_path']).expanduser()
        if not bag.is_absolute():bag=root/bag
        if not bag.is_file():raise FileNotFoundError(bag)
        label=category_for(sid,cfg)
        rows=index_bag(bag,sid,label,topic,args.save_every_n,args.max_frames)
        allrows.extend(rows)
        print(f'[indexed] scenario={sid} label={label} rows={len(rows)} splits={dict(Counter(r["split"] for r in rows))}',flush=True)
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(allrows)
    print(f'[saved] {output}, {len(allrows)} image records, {dict(Counter(r["split"] for r in allrows))}')
    print('[note] 4:1 splits are within-scenario and temporally correlated. Use held-out sessions for unbiased generalization testing.')

if __name__=='__main__':main()
