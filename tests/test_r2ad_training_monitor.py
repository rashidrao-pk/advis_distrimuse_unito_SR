import json
import sys

import cv2
import numpy as np
import torch
import yaml
import pytest
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


@pytest.mark.parametrize('workers', [0, 1])
def test_training_reports_and_transposed_random_probes(tmp_path, monkeypatch, workers):
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
        '--output', str(output), '--epochs', '1', '--batch-size', '1', '--workers', str(workers), '--device', 'cpu',
        '--latent-dims', '2', '--patch-size', '16', '--samples-per-frame', '1',
        '--monitor-every', '1', '--monitor-probes', '1'])
    training.main()
    summary = json.loads((output.parent / 'training_summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['precision'] == 'fp32'
    assert summary['optimizer_steps'] == 1
    assert summary['best_epoch'] == 1
    assert summary['estimated_remaining_seconds'] == 0
    assert summary['elapsed_seconds'] >= summary['epochs'][0]['epoch_seconds']
    assert (output.parent / 'epoch_metrics.csv').is_file()
    assert cv2.imread(str(output.parent / 'training_progress.png')) is not None
    epoch = summary['epochs'][0]
    assert 0 <= epoch['discriminator_accuracy'] <= 1
    assert 0 <= epoch['discriminator_f1'] <= 1
    assert epoch['weighted_kl'] == pytest.approx(epoch['kl'] * .0001)
    assert epoch['weighted_gan'] == pytest.approx(epoch['generator_adversarial'] * .01)
    assert 'normal training frames' in (output.parent / 'training_report.md').read_text()
    assert output.is_file()
    monitor = output.parent / 'live_monitor'
    assert (monitor / 'fixed_probes.pt').is_file()
    grid = cv2.imread(str(monitor / 'probe_step_000001.png'))
    assert grid is not None
    # One fixed + two random columns, three rows: square rather than old tall layout.
    assert grid.shape[0] == grid.shape[1]
    with (monitor / 'batch_losses.csv').open() as stream:
        assert len(list(__import__('csv').DictReader(stream))) == 1


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


def test_epoch_discriminator_metrics():
    result = training.epoch_learning_metrics(torch.tensor([2.,4.,6.,8.,10.]),
                                            torch.tensor([3.,1.,3.,1.]), 2, .1, .2)
    assert result['discriminator_accuracy'] == .75
    assert result['discriminator_f1'] == .75
    assert result['weighted_kl'] == pytest.approx(.2)
    assert result['weighted_gan'] == pytest.approx(.6)


def test_error_maps_use_fixed_red_scale(tmp_path):
    from matplotlib import colormaps
    from r2ad.core import save_map
    frame = np.zeros((2, 2, 3), np.uint8)
    mask = np.ones((2, 2), np.uint8)
    values = np.array([[0., .1], [.25, .5]], np.float32)
    for name, last in [('first', .5), ('second', 1.)]:
        amap = values.copy()
        amap[1, 1] = last
        out = tmp_path / f'{name}.png'
        save_map(out, frame, mask, {'map': amap, 'covered': mask.astype(bool)})
        heat = cv2.imread(str(tmp_path / f'{name}_heat.png'))
        expected = colormaps['Reds'](np.clip(amap / .5, 0, 1), bytes=True)[..., :3][..., ::-1]
        assert np.array_equal(heat, expected)
        assert np.array_equal(np.load(out.with_suffix('.npy')), amap)
        scale = json.loads((tmp_path / f'{name}_scale.json').read_text())
        assert scale == {'cmap': 'Reds', 'vmin': 0., 'vmax': .5}
    assert np.array_equal(cv2.imread(str(tmp_path / 'first_heat.png'))[0],
                          cv2.imread(str(tmp_path / 'second_heat.png'))[0])
