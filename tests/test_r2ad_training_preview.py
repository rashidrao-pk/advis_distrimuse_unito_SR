import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

from r2ad.dataset import RectangleDataset
from r2ad.training_preview import save_training_preview
from scripts import train_r2ad


def inputs(tmp_path):
    mask=tmp_path/'mask.png'
    roi=np.zeros((64,96),np.uint8);roi[8:56,8:88]=255
    cv2.imwrite(str(mask),roi)
    for name in ('train','val'):
        folder=tmp_path/name;folder.mkdir()
        cv2.imwrite(str(folder/'frame.png'),np.full((64,96,3),120,np.uint8))
    return mask


def test_preview_preserves_native_size_rng_and_original_artifacts(tmp_path):
    mask=inputs(tmp_path)
    settings=dict(aug_translate=.005,aug_rotate=0,aug_zoom_min=1,aug_zoom_max=1.03,
                  patch_scale_min=1,patch_scale_max=1)
    train=RectangleDataset(tmp_path/'train',mask,patch=32,augmentation_settings=settings)
    val=RectangleDataset(tmp_path/'val',mask,patch=32,augment=False,augmentation_settings=settings)
    args=SimpleNamespace(mask=str(mask),safety_area='SafetyZone',patch_size=32,fixed_patch_size=True,
                         min_coverage=.85,samples_per_frame=6,preview_samples=2,seed=42)
    np.random.seed(3);before=np.random.get_state();torch_before=torch.get_rng_state().clone()
    directory,report=save_training_preview(train,val,args,tmp_path/'setup')
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert torch.equal(torch_before,torch.get_rng_state())
    assert report['source_patch_size']==32 and report['model_input_size']==128
    assert report['stage']=='before_first_optimizer_update'
    assert len(report['samples'])==4
    assert cv2.imread(str(directory/'train_01_source_crop.png')).shape==(32,32,3)
    assert cv2.imread(str(directory/'train_01_model_input.png')).shape==(128,128,3)
    assert np.load(directory/'train_01_model_input.npy').shape==(3,128,128)
    assert all(r['patch_mask_occupancy']>=.85 for r in report['samples'])
    assert all(not r['augmentation'] for r in report['samples'] if r['split']=='val')
    assert (directory/'index.html').is_file() and (directory/'train_samples.jpg').is_file()
    original=(directory/'plan.json').read_bytes()
    resumed,_=save_training_preview(train,val,args,directory,resuming=True)
    assert resumed!=directory and (directory/'plan.json').read_bytes()==original


def test_preview_only_never_constructs_model_or_writes_run_state(tmp_path,monkeypatch):
    mask=inputs(tmp_path)
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'data':{'dataset_base':str(tmp_path)},'r2ad':{
        'safety_zone_mask':str(mask),'train_frames':str(tmp_path/'train'),'val_frames':str(tmp_path/'val')}}))
    def forbidden(*args):raise AssertionError('Models must not be constructed in preview-only mode')
    monkeypatch.setattr(train_r2ad,'build_models',forbidden)
    output=tmp_path/'run'/'model.pt'
    monkeypatch.setattr(sys,'argv',['train','--config',str(config),'--safety-area','SafetyZone',
        '--output',str(output),'--patch-size','32','--fixed-patch-size','--preview-only','--preview-samples','1',
        '--device','cuda','--precision','bf16'])
    train_r2ad.main()
    report=json.loads((output.parent/'training_setup'/'plan.json').read_text())
    assert report['stage']=='preview_only_no_training_performed'
    assert not output.exists()
    assert not (output.parent/'training_summary.json').exists()
    assert not (output.parent/'last_checkpoint.pt').exists()
