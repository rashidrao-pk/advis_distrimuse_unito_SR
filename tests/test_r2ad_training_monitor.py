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
        '--monitor-every', '1', '--monitor-probes', '1', '--save-cp-k', '1'])
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
    periodic = output.parent / 'checkpoints' / 'model_r2ad_epoch_0001.pt'
    assert periodic.is_file()
    saved = torch.load(periodic, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 1 and saved['config']['save_cp_k'] == 1
    assert epoch['periodic_checkpoint'] == str(periodic)
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


@pytest.mark.parametrize('stop_after',[1,3])
def test_mid_epoch_resume_matches_uninterrupted_and_legacy_warm_start(tmp_path,monkeypatch,stop_after):
    from types import SimpleNamespace
    mask=tmp_path/'mask.png'
    cv2.imwrite(str(mask),np.full((32,32),255,np.uint8))
    for split in ('train','val'):
        folder=tmp_path/split;folder.mkdir()
        for i in range(3):
            cv2.imwrite(str(folder/f'frame_{i}.png'),np.full((32,32,3),80+i*30,np.uint8))
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'data':{'dataset_base':str(tmp_path)},'r2ad':{
        'safety_zone_mask':str(mask),'train_frames':str(tmp_path/'train'),'val_frames':str(tmp_path/'val')}}))
    monkeypatch.setattr(training,'build_models',lambda *args:(SmallEncoder(),SmallDecoder(),SmallDiscriminator()))
    monkeypatch.setattr(training,'save_training_dashboard',lambda *args:None)
    base=['train','--config',str(config),'--safety-area','SafetyZone','--epochs','2','--batch-size','2',
          '--workers','0','--device','cpu','--latent-dims','2','--patch-size','16',
          '--samples-per-frame','2','--monitor-every','0','--save-every-steps','1','--seed','7']
    baseline=tmp_path/'baseline'/'model.pt'
    monkeypatch.setattr(sys,'argv',base+['--output',str(baseline)])
    training.main()
    expected=torch.load(baseline.parent/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    pause=SimpleNamespace(requested=False)
    real_adam=torch.optim.Adam
    optimizers=[]
    updates=[]
    def adam(*args,**kwargs):
        optimizer=real_adam(*args,**kwargs)
        if not optimizers:
            original=optimizer.step
            def step(*a,**kw):
                result=original(*a,**kw);updates.append(1)
                if len(updates)==stop_after:pause.requested=True
                return result
            optimizer.step=step
        optimizers.append(optimizer)
        return optimizer
    monkeypatch.setattr(training.torch.optim,'Adam',adam)
    interrupted=tmp_path/'paused'/'model.pt'
    monkeypatch.setattr(sys,'argv',base+['--output',str(interrupted)])
    training.train(pause)
    last=interrupted.parent/'last_checkpoint.pt'
    stopped=torch.load(last,map_location='cpu',weights_only=False)
    assert stopped['training_state']['next_batch']==stop_after
    assert stopped['training_state']['global_step']==stop_after
    assert stopped['training_state']['summary']['status']=='paused'
    assert stopped['optimizer_g_state_dict']['state']
    monkeypatch.setattr(training.torch.optim,'Adam',real_adam)
    monkeypatch.setattr(sys,'argv',['train','--resume',str(last),'--epochs','2'])
    training.main()
    actual=torch.load(last,map_location='cpu',weights_only=False)
    assert actual['training_state']['next_epoch']==3
    assert actual['training_state']['global_step']==6
    for name in ('encoder_state_dict','decoder_state_dict','discriminator_state_dict'):
        for key in expected[name]:
            assert torch.equal(expected[name][key],actual[name][key]),(name,key)
    assert len(actual['training_state']['summary']['epochs'])==2
    # Completed-epoch resume advances to the next epoch and carries optimizer/history state.
    monkeypatch.setattr(sys,'argv',['train','--resume',str(last),'--epochs','3'])
    training.main()
    third=torch.load(last,map_location='cpu',weights_only=False)
    assert third['training_state']['next_epoch']==4
    assert third['training_state']['global_step']==9
    assert len(third['training_state']['summary']['epochs'])==3
    cv2.imwrite(str(mask),np.zeros((32,32),np.uint8))
    monkeypatch.setattr(sys,'argv',['train','--resume',str(last),'--epochs','4'])
    with pytest.raises(SystemExit):training.main()
    cv2.imwrite(str(mask),np.full((32,32),255,np.uint8))
    # Older snapshots remain usable as explicit, reported warm starts.
    legacy=tmp_path/'legacy.pt'
    torch.save({k:v for k,v in actual.items() if k not in
                ('training_state','optimizer_g_state_dict','optimizer_d_state_dict','best_val_mse')},legacy)
    warm=tmp_path/'warm'/'model.pt'
    monkeypatch.setattr(sys,'argv',['train','--resume',str(legacy),'--epochs','3','--output',str(warm),
                                  '--aug-translate','.005','--aug-rotate','.5','--patch-scale-min','1','--patch-scale-max','1'])
    training.main()
    continued=torch.load(warm.parent/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert continued['epoch']==3
    assert continued['training_state']['global_step']==9
    assert continued['config']['aug_translate']==.005
    assert continued['config']['aug_rotate']==.5
    assert continued['training_state']['summary']['augmentation_change']['current']['patch_scale_min']==1


def test_pause_signals_restore_handlers():
    import signal
    from r2ad.training_state import PauseSignals
    before=signal.getsignal(signal.SIGINT)
    with PauseSignals() as pause:
        pause.request()
        assert pause.requested
        with pytest.raises(KeyboardInterrupt):pause.request()
    assert signal.getsignal(signal.SIGINT)==before


def test_atomic_checkpoint_failure_preserves_previous(tmp_path,monkeypatch):
    from r2ad import training_state
    path=tmp_path/'last_checkpoint.pt';path.write_bytes(b'previous checkpoint')
    def broken_save(value,temporary):
        temporary.write_bytes(b'incomplete')
        raise OSError('simulated failed write')
    monkeypatch.setattr(training_state.torch,'save',broken_save)
    with pytest.raises(OSError):training_state.atomic_save({},path)
    assert path.read_bytes()==b'previous checkpoint'
