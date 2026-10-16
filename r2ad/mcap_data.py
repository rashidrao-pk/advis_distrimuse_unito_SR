"""Indexed ROS2 MCAP loading for R2AD. No ROS installation is required.

Uses MCAP indexed time-range reads. Reader/file handles are worker-local.
"""
import csv
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset

from .core import read_mask, random_boxes, extract

FIELDS = ('scenario','label','split','bag','topic','log_time','occurrence','ordinal')


def load_manifest(path, split=None):
    with open(path, newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r['split'] == split]
    if not rows:
        raise ValueError(f'Manifest has no rows for split={split!r}: {path}')
    return rows


def decode_image(msg, schema_name):
    typ = schema_name.replace('/msg/', '/')
    if typ.endswith('/CompressedImage'):
        image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode JPEG/PNG CompressedImage')
        return image
    if not typ.endswith('/Image'):
        raise ValueError(f'Unsupported ROS type: {schema_name}')
    encoding = str(msg.encoding).lower()
    mapping = {'bgr8': (3, None), 'rgb8': (3, cv2.COLOR_RGB2BGR),
               'bgra8': (4, cv2.COLOR_BGRA2BGR), 'rgba8': (4, cv2.COLOR_RGBA2BGR),
               'mono8': (1, cv2.COLOR_GRAY2BGR), '8uc1': (1, cv2.COLOR_GRAY2BGR)}
    if encoding not in mapping:
        raise ValueError(f'Unsupported ROS encoding {encoding}')
    channels, convert = mapping[encoding]
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    required = w * channels
    if step < required:
        raise ValueError('Malformed image row step')
    buffer = np.frombuffer(msg.data, dtype=np.uint8)
    if len(buffer) < step * h:
        raise ValueError('Truncated image payload')
    frame = np.ascontiguousarray(buffer[:step*h].reshape(h, step)[:, :required]).reshape(h, w, channels)
    if channels == 1:
        frame = frame[:, :, 0]
    return cv2.cvtColor(frame, convert) if convert is not None else frame


class McapFrameReader:
    """Seek messages by nanosecond timestamp; keep a small per-process set of readers.

    Requires MCAP chunk indexes for efficient access. Timestamp ties resolved by
    'occurrence', the index among matching topic+log_time messages.
    """
    def __init__(self, max_open_bags=2):
        self.max_open_bags = max_open_bags
        self._readers = OrderedDict()

    def __getstate__(self):
        return {'max_open_bags': self.max_open_bags, '_readers': OrderedDict()}

    def close(self):
        while self._readers:
            _, (handle, _) = self._readers.popitem(last=False)
            handle.close()

    def __del__(self):
        self.close()

    def _reader(self, bag):
        bag = str(bag)
        if bag in self._readers:
            self._readers.move_to_end(bag)
            return self._readers[bag][1]
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
        while len(self._readers) >= self.max_open_bags:
            _, (old, _) = self._readers.popitem(last=False)
            old.close()
        handle = open(bag, 'rb')
        try:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            self._readers[bag] = (handle, reader)
            return reader
        except Exception:
            handle.close()
            raise

    def read(self, record):
        bag, topic = record['bag'], record['topic']
        stamp, nth = int(record['log_time']), int(record['occurrence'])
        reader = self._reader(bag)
        found = 0
        for schema, channel, _, decoded in reader.iter_decoded_messages(
                topics=[topic], start_time=stamp, end_time=stamp + 1):
            if found == nth:
                return decode_image(decoded, schema.name)
            found += 1
        raise IndexError(f'Frame not found: {bag}, {topic}, ns={stamp}, occurrence={nth}')


class McapRectangleDataset(Dataset):
    def __init__(self, manifest, split, mask, patch=224, coverage=.85,
                 samples_per_frame=6, augment=True, seed=42, max_open_bags=2):
        if split not in ('train', 'val'):
            raise ValueError('Expected train or val split')
        self.rows = load_manifest(manifest, split)
        if any(r['label'] != 'normal' for r in self.rows):
            raise ValueError('Training/validation must be normal-only')
        self.mask_path = str(mask)
        self.patch, self.coverage = patch, coverage
        self.samples_per_frame = samples_per_frame
        self.augment, self.seed = augment, seed
        self.reader = McapFrameReader(max_open_bags)
        self._mask_cache = {}

    def __len__(self):
        return len(self.rows) * self.samples_per_frame

    def __getitem__(self, index):
        record = self.rows[index // self.samples_per_frame]
        frame = self.reader.read(record)
        key = frame.shape[:2]
        if key not in self._mask_cache:
            self._mask_cache[key] = read_mask(self.mask_path, frame.shape)
        mask = self._mask_cache[key]
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        if self.augment:
            h,w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w/2,h/2),float(rng.uniform(-2,2)),1.)
            M[:,2] += (float(rng.uniform(-.03,.03)*w),float(rng.uniform(-.03,.03)*h))
            frame = cv2.warpAffine(frame,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask,M,(w,h),flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT)
            frame = np.uint8(np.clip(frame.astype(np.float32)*rng.uniform(.85,1.15)
                                     + rng.uniform(-12,12),0,255))
        box = random_boxes(mask,self.patch,1,rng,self.coverage)[0]
        return extract(frame,box)
