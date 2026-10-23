import json
import sys

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from scripts import train_r2ad as training


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 4)

    def forward(self, x):
        return self.layer(x.mean((2, 3))).chunk(2, dim=1)


class SmallDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3)

    def forward(self, x):
        return self.layer(x).tanh()[:, :, None, None].expand(-1, -1, 128, 128)


class SmallDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 1)

    def forward(self, x):
        return self.layer(x.mean((2, 3)))


def test_training_reports_and_transposed_random_probes(tmp_path, monkeypatch):
    mask = tmp_path / 'mask.png'
    cv2.imwrite(str(mask), np.full((32, 32), 255, np.uint8))
    for split in ('train', 'val'):
        folder = tmp_path / split
        folder.mkdir()
        cv2.imwrite(str(folder / 'frame.png'), np.full((32, 32, 3), 128, np.uint8))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': {'dataset_base': str(tmp_path)}, 'r2ad': {
        'safety_zone_mask': str(mask), 'train_frames': str(tmp_path / 'train'),
        'val_frames': str(tmp_path / 'val')}}))
    monkeypatch.setattr(training, 'build_models', lambda *args: (SmallEncoder(), SmallDecoder(), SmallDiscriminator()))
    output = tmp_path / 'run' / 'model_r2ad.pt'
    monkeypatch.setattr(sys, 'argv', ['train_r2ad.py', '--config', str(config), '--safety-area', 'SafetyZone',
        '--output', str(output), '--epochs', '1', '--batch-size', '1', '--workers', '0', '--device', 'cpu',
        '--latent-dims', '2', '--patch-size', '16', '--samples-per-frame', '1',
        '--monitor-every', '1', '--monitor-probes', '1'])
    training.main()
    summary = json.loads((output.parent / 'training_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['optimizer_steps'] == 1
    assert summary['best_epoch'] == 1
    assert summary['estimated_remaining_seconds'] == 0
    assert summary['elapsed_seconds'] >= summary['epochs'][0]['epoch_seconds']
    assert (output.parent / 'epoch_metrics.csv').is_file()
    assert 'normal training frames' in (output.parent / 'training_report.md').read_text()
    assert output.is_file()
    monitor = output.parent / 'live_monitor'
    assert (monitor / 'fixed_probes.pt').is_file()
    grid = cv2.imread(str(monitor / 'probe_step_000001.png'))
    assert grid is not None
    # One fixed + two random columns, three rows: square rather than old tall layout.
    assert grid.shape[0] == grid.shape[1]


def test_random_monitor_does_not_change_training_numpy_rng():
    class RandomDataset:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.tensor(np.random.random((3, 2, 2)))
    np.random.seed(10)
    state = np.random.get_state()
    first = training.random_monitor_patches(RandomDataset(), 2, 42)
    actual = np.random.random()
    np.random.set_state(state)
    assert actual == np.random.random()
    assert torch.equal(first, training.random_monitor_patches(RandomDataset(), 2, 42))
    assert not torch.equal(first, training.random_monitor_patches(RandomDataset(), 2, 43))
