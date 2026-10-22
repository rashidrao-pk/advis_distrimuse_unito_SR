import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from r2ad.core import images
from r2ad.dataset import RectangleDataset
from r2ad.frame_manifest import read_frame_manifest


def test_manifest_split_end_to_end(tmp_path):
    frame = np.full((16, 16, 3), 128, dtype=np.uint8)
    for label, scenarios in [('normal', range(5)), ('unexpected', range(1))]:
        for scenario in scenarios:
            folder = tmp_path / label / f'scenario_{scenario}'
            folder.mkdir(parents=True)
            for index in range(2):
                assert cv2.imwrite(str(folder / f'frame_{index}.png'), frame)
    mask = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask), np.full((16, 16), 255, dtype=np.uint8))
    output = tmp_path / 'splits'
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'data': {'dataset_base': str(tmp_path), 'mask_types': {
            area: str(mask) for area in ('PLeft', 'PRight', 'RoboArm', 'ConvBelt')}},
        'r2ad': {'normal_source': str(tmp_path / 'normal'),
                 'unexpected_source': str(tmp_path / 'unexpected'),
                 'preprocessed_dir': str(output)},
    }))
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(repo / 'scripts/preprocess_r2ad.py'),
                    '--config', str(config)], cwd=repo, check=True, capture_output=True, text=True)
    assert {p.name for p in output.iterdir()} == {
        'train.csv', 'val.csv', 'test.csv', 'normal_test.csv', 'manifest.csv', 'metadata.json'}
    train, val, test = [set(images(output / f'{name}.csv')) for name in ('train', 'val', 'test')]
    assert len(train) == 6 and len(val) == 2 and len(test) == 4
    assert not (train & val or train & test or val & test)
    assert {p.parent for p in train}.isdisjoint(p.parent for p in val | test)
    assert len(images(output / 'normal_test.csv')) == 2
    assert all(p.parent.parent.name == 'normal' for p in train | val)
    with (output / 'manifest.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and all(row['processed'] == '' for row in rows)
    dataset = RectangleDataset(output / 'train.csv', mask, patch=8,
                               samples_per_frame=1, augment=False)
    # Existing network input conversion resizes sampled patches to 128x128.
    assert tuple(dataset[0].shape) == (3, 128, 128)


def test_manifest_relative_missing_duplicate_and_invalid(tmp_path):
    source = tmp_path / 'frame.png'
    source.write_bytes(b'placeholder')
    manifest = tmp_path / 'train.csv'
    manifest.write_text('source\nframe.png\n')
    assert read_frame_manifest(manifest) == [source]
    manifest.write_text('source\nframe.png\nframe.png\n')
    with pytest.raises(ValueError, match='Duplicate'):
        read_frame_manifest(manifest)
    source.unlink()
    with pytest.raises(FileNotFoundError, match='Missing source'):
        read_frame_manifest(manifest)
    manifest.write_text('bag,split\nx,train\n')
    with pytest.raises(ValueError, match='source column'):
        read_frame_manifest(manifest)
