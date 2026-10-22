#!/usr/bin/env python3
"""Export original-size ROS2 MCAP camera frames into R2AD scenario folders.

No ROS2, rclpy, rosbag2_py or cv_bridge required. Python packages:
  pip install mcap mcap-ros2-support pyyaml opencv-python-headless numpy
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def frame_counts(reader, topics):
    """Count source frames without decoding images; prefer the MCAP index."""
    counts = dict.fromkeys(topics, 0)
    summary = reader.get_summary()
    if summary is not None and summary.statistics is not None:
        for cid, count in summary.statistics.channel_message_counts.items():
            topic = summary.channels[cid].topic
            if topic in counts:
                counts[topic] += count
    else:
        for _, channel, _ in reader.iter_messages(topics=topics):
            counts[channel.topic] += 1
    return counts


class FrameProgress:
    """Throttled terminal progress, also usable in redirected job logs."""

    def __init__(self, sid, counts, seen, saved, failures):
        self.sid, self.counts = sid, counts
        self.seen, self.saved, self.failures = seen, saved, failures
        self.started = time.monotonic()
        self.updated = 0
        self.width = 0
        self.terminal = sys.stderr.isatty()

    def show(self, topic=None, index=None, final=False):
        now = time.monotonic()
        if not final and now - self.updated < (0.2 if self.terminal else 5):
            return
        self.updated = now
        done, total = sum(self.seen.values()), sum(self.counts.values())
        elapsed = now - self.started
        rate = done / elapsed if elapsed else 0
        fraction = min(done / total, 1) if total else 0
        filled = int(24 * fraction)
        eta = time.strftime('%H:%M:%S', time.gmtime((total - done) / rate)) if rate else '--:--:--'
        current = f' {topic} frame_index={index}' if topic is not None else ''
        line = (f'Scenario {self.sid} [{"#" * filled}{"-" * (24 - filled)}] '
                f'{fraction:6.1%} source_frames={done}/{total}{current} '
                f'saved={sum(self.saved.values())} errors={sum(self.failures.values())} '
                f'{rate:.1f} frames/s elapsed={elapsed:.0f}s ETA={eta}')
        if final and done < total:
            line += ' (stopped before end of bag)'
        print(('\r' if self.terminal else '') + line.ljust(self.width),
              end='\n' if final or not self.terminal else '', file=sys.stderr, flush=True)
        self.width = len(line)


def category_for(scenario, config):
    idx = int(scenario.split('_')[0])
    for k, value in (config.get('scenario_categories') or {}).items():
        match = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', str(k))
        if match and int(match.group(1)) <= idx <= int(match.group(2) or match.group(1)):
            label = str(value).lower()
            if label in ('normal', 'expected'):
                return 'normal'
            if label in ('anomalous', 'anomaly', 'abnormal', 'unexpected'):
                return 'unexpected'
            raise ValueError(f'Unknown scenario category: {value}')
    raise ValueError(f'Scenario {scenario} missing from scenario_categories')


def resolve_root(config, kind, override):
    base = Path(config['data']['dataset_base']).expanduser()
    if override:
        return Path(override).expanduser() / kind
    value = (config.get('r2ad') or {}).get('normal_source' if kind == 'normal' else 'unexpected_source')
    if value and 'REPLACE_ME' not in str(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else base / path
    return base / 'r2ad' / 'sources' / kind


def decode_ros_image(msg, msg_type):
    """Decode common ROS sensor_msgs Image/CompressedImage types without cv_bridge."""
    typ = msg_type.replace('/msg/', '/')
    if typ.endswith('/CompressedImage'):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Compressed image cannot be decoded')
        return image
    if not typ.endswith('/Image'):
        raise ValueError(f'Unsupported message type: {msg_type}')
    encoding = str(msg.encoding).lower()
    encodings = {
        'bgr8': (np.uint8, 3, None),
        'rgb8': (np.uint8, 3, cv2.COLOR_RGB2BGR),
        'bgra8': (np.uint8, 4, cv2.COLOR_BGRA2BGR),
        'rgba8': (np.uint8, 4, cv2.COLOR_RGBA2BGR),
        'mono8': (np.uint8, 1, cv2.COLOR_GRAY2BGR),
        '8uc1': (np.uint8, 1, cv2.COLOR_GRAY2BGR),
        'mono16': (np.uint16, 1, None),
        '16uc1': (np.uint16, 1, None),
    }
    if encoding not in encodings:
        raise ValueError(f'Unsupported Image encoding {encoding!r}')
    dtype, channels, conversion = encodings[encoding]
    if dtype == np.uint16:
        dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
    else:
        dtype = np.dtype(dtype)
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    required = width * channels * dtype.itemsize
    if step < required:
        raise ValueError(f'Image row stride {step} smaller than {required}')
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < step * height:
        raise ValueError('Truncated Image message')
    rows = data[:step * height].reshape(height, step)
    valid = np.ascontiguousarray(rows[:, :required])
    decoded = valid.view(dtype).reshape(height, width, channels)
    if channels == 1:
        decoded = decoded[:, :, 0]
    if conversion is not None:
        decoded = cv2.cvtColor(decoded, conversion)
    return decoded


def run_scenario(config, sid, args):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    opts = config['scenario_options']
    bag_path = Path(opts[sid]['rosbag_path']).expanduser()
    if not bag_path.is_absolute():
        bag_path = Path(config['data']['dataset_base']).expanduser() / bag_path
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    group = args.category if args.category != 'auto' else category_for(sid, config)
    out_root = resolve_root(config, group, args.output_dir)
    out_dir = out_root / f'scenario_{sid}'
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = [c if c.startswith('/') else f'/camera/{c}/image_raw' for c in args.camera]
    print(f'[R2AD] scenario={sid} class={group} bag={bag_path}')
    print(f'[R2AD] topics={topics} output={out_dir}')
    seen = {t: 0 for t in topics}
    saved = {t: 0 for t in topics}
    failures = {t: 0 for t in topics}
    last_shape = {}
    with bag_path.open('rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        progress = None
        if args.progress:
            print(f'[R2AD] scenario={sid}: counting source frames...', flush=True)
            counts = frame_counts(reader, topics)
            print(f'[R2AD] source frames per topic: {counts}', flush=True)
            progress = FrameProgress(sid, counts, seen, saved, failures)
            progress.show()
        for schema, channel, message, decoded in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            if topic not in seen:
                continue
            index = seen[topic]
            seen[topic] += 1
            if progress:
                progress.show(topic, index)
            if index % args.save_every_n:
                continue
            if args.max_frames and saved[topic] >= args.max_frames:
                if all(saved[t] >= args.max_frames for t in topics):
                    break
                continue
            filename_prefix = 'frame' if len(topics) == 1 else topic.split('/')[2] + '_frame'
            path = out_dir / f'{filename_prefix}_{index:06d}.{args.image_format}'
            try:
                if path.exists() and not args.overwrite:
                    raise FileExistsError(f'{path} exists; use --overwrite')
                image = decode_ros_image(decoded, schema.name)
                last_shape[topic] = image.shape
                params = ([cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
                          if args.image_format == 'jpg' else [])
                if not cv2.imwrite(str(path), image, params):
                    raise IOError(f'Failed saving {path}')
                saved[topic] += 1
            except Exception as exc:
                failures[topic] += 1
                print(f'\n[ERROR] {sid} {topic} frame {index}: {exc}', file=sys.stderr)
            if args.max_frames and all(saved[t] >= args.max_frames for t in topics):
                break
        if progress:
            progress.show(final=True)
    metadata_root = out_root / '_metadata'
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata = {'scenario': sid, 'category': group, 'bag': str(bag_path),
                'camera_topics': topics, 'seen': seen, 'saved': saved,
                'failures': failures, 'frame_shape_hwc': last_shape,
                'source_resolution_preserved': True, 'save_every_n': args.save_every_n}
    (metadata_root / f'scenario_{sid}.json').write_text(json.dumps(metadata, indent=2))
    print(f'[R2AD] completed {sid}: saved={saved} failures={failures} shapes={last_shape}')
    if not sum(saved.values()) or any(failures.values()):
        return False
    return True


def main():
    p = argparse.ArgumentParser(description='ROS-free R2AD full-frame extraction from MCAP.')
    p.add_argument('--config', default='configs/cf_dataset_epito.yaml')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--scenario', nargs='+')
    group.add_argument('--all', action='store_true')
    p.add_argument('--camera', nargs='+')
    p.add_argument('--process-to', choices=['frames'], default='frames')
    p.add_argument('--category', choices=['auto', 'normal', 'unexpected'], default='auto')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--save-every-n', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--image-format', choices=['png', 'jpg'], default='png')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--progress', action='store_true',
                   help='Show source-frame progress, current frame index, saved/errors, speed and ETA.')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--target-size', type=int, default=None,
                   help='Accepted for CLI compatibility but ignored (original resolution retained).')
    args = p.parse_args()
    if args.save_every_n < 1 or (args.max_frames is not None and args.max_frames < 1):
        p.error('save-every-n and max-frames must be positive')
    if args.target_size:
        print('[R2AD] ignoring --target-size; preserving native camera resolution')
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)
    ids = (list(config['scenario_options']) if args.all else
           args.scenario or [str(config['scenario']['id'])])
    unknown = [sid for sid in ids if sid not in config['scenario_options']]
    if unknown:
        p.error(f'Unknown scenario IDs: {unknown}')
    args.camera = args.camera or [config.get('playback_options', {}).get('camera', 'back_view')]
    ok = True
    for scenario_number, sid in enumerate(ids, 1):
        print(f'[R2AD] scenario {scenario_number}/{len(ids)}: {sid}', flush=True)
        try:
            ok = run_scenario(config, sid, args) and ok
        except Exception as exc:
            print(f'[ERROR] scenario={sid}: {exc}', file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
