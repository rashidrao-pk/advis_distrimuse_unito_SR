import csv
import json
import sys

import numpy as np
import pytest
import cv2
import torch
import yaml

from r2ad.core import checkpoint_patch_settings
from r2ad.latent_audit import latent_statistics,save_latent_audit


def test_collapsed_and_redundant_latents():
    mu=np.zeros((20,4));logvar=np.zeros_like(mu)
    stats,dimensions,_=latent_statistics(mu,logvar)
    assert stats['active_units']==0
    assert stats['total_mean_kl_nats']==0
    assert stats['effective_rank_participation']==0
    t=np.linspace(-2,2,20)
    mu=np.column_stack([t,2*t,np.zeros_like(t),np.zeros_like(t)])
    stats,dimensions,_=latent_statistics(mu,logvar)
    assert stats['active_units']==2
    assert stats['effective_rank_participation']==pytest.approx(1)
    assert stats['principal_components_for_95_percent_variance']==1
    assert stats['mean_kl_per_dimension_nats']==pytest.approx(stats['total_mean_kl_nats']/4)
    assert dimensions[2]['active'] is False


def test_invalid_samples_and_files(tmp_path):
    with pytest.raises(ValueError):latent_statistics(np.zeros((1,4)),np.zeros((1,4)))
    with pytest.raises(ValueError):latent_statistics(np.zeros((4,4)),np.zeros((4,3)))
    rng=np.random.default_rng(3)
    summary=save_latent_audit(rng.normal(size=(40,8)),np.zeros((40,8)),tmp_path,{'scope':'unit_test'})
    assert summary['latent_dims']==8
    assert (tmp_path/'dimensions.csv').is_file()
    assert (tmp_path/'latent_utilization.png').is_file()


def test_inference_patch_size_tracks_checkpoint():
    assert checkpoint_patch_settings({'config':{'patch_size':128}})==(128,32)
    assert checkpoint_patch_settings({'config':{'patch_size':224}})==(224,64)
    assert checkpoint_patch_settings({})==(224,64)
    assert checkpoint_patch_settings({},32,16)==(32,16)
    with pytest.raises(ValueError,match='mismatch'):
        checkpoint_patch_settings({'config':{'patch_size':128}},224)


def test_audit_cli_uses_validation_and_checkpoint_patch_size(tmp_path,monkeypatch):
    from scripts import audit_r2ad_latents as audit
    class Encoder(torch.nn.Module):
        def forward(self,x):
            mu=x.mean((2,3));return mu,torch.zeros_like(mu)
    mask=tmp_path/'mask.png'
    cv2.imwrite(str(mask),np.full((48,48),255,np.uint8))
    rows=[]
    for i in range(2):
        path=tmp_path/f'frame_{i}.png'
        cv2.imwrite(str(path),np.full((48,48,3),80+i*60,np.uint8))
        rows.append(dict(source=str(path),label='normal',split='val'))
    manifest=tmp_path/'val.csv'
    with manifest.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'r2ad':{'safety_zone_mask':str(mask)}}))
    monkeypatch.setattr(audit,'load_models',lambda *args:(Encoder(),torch.nn.Identity(),{
        'epoch':3,'config':{'patch_size':32,'fixed_patch_size':True,'safety_area':'SafetyZone'}}))
    output=tmp_path/'audit'
    monkeypatch.setattr(sys,'argv',['audit','--config',str(config),'--checkpoint',str(tmp_path/'model.pt'),
        '--input',str(manifest),'--output-dir',str(output),'--samples','8','--batch-size','3','--device','cpu'])
    audit.main()
    report=json.loads((output/'summary.json').read_text())
    assert report['samples']==8 and report['latent_dims']==3
    assert report['metadata']['patch_size']==32
    assert report['metadata']['patch_scale']==[1.,1.]
    assert report['metadata']['unique_frames']==2


def test_wider_latent_preserves_network_resolution():
    from r2ad.core import Encoder,Decoder
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        encoder=Encoder(128);decoder=Decoder(128)
        mu,logvar=encoder(torch.zeros(1,3,128,128))
        assert mu.shape==logvar.shape==(1,128)
        assert decoder(mu).shape==(1,3,128,128)
