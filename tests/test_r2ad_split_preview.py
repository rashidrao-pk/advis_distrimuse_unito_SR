import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def test_split_previews_and_leakage(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    mask = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask), np.full((48, 64), 255, np.uint8))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': {'dataset_base': str(tmp_path)},
                                     'r2ad': {'safety_zone_mask': str(mask)}}))
    for split, scenario, label in [('train', '1_0', 'normal'), ('val', '2_0', 'normal'), ('test', '8_0', 'anomalous')]:
        folder = tmp_path / f'scenario_{scenario}'
        folder.mkdir()
        source = folder / 'frame_000000.png'
        assert cv2.imwrite(str(source), np.full((48, 64, 3), 128, np.uint8))
        with (tmp_path / f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source', 'scenario', 'split', 'label', 'label_PLeft'])
            writer.writeheader()
            writer.writerow(dict(source=str(source), scenario=scenario, split=split, label=label, label_PLeft=label))
    output = tmp_path / 'preview'
    command = [sys.executable, str(repo / 'scripts/preview_r2ad_splits.py'), '--config', str(config),
               '--split-dir', str(tmp_path), '--output-dir', str(output), '--samples', '2']
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads((output / 'report.json').read_text())
    assert not report['issues']
    assert report['samples']['test_anomalous'][0]['scenario'] == '8_0'
    for name in ('train_normal', 'val_normal', 'test_anomalous'):
        assert cv2.imread(str(output / f'{name}.jpg')) is not None
    assert (output / 'index.html').is_file()
    # A repeated scenario must fail even if each split has different images.
    val = tmp_path / 'val.csv'
    val.write_text(val.read_text().replace(',2_0,', ',1_0,'))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'shared scenarios' in result.stdout
