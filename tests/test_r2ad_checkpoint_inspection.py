import csv
import json
import sys

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from r2ad.core import infer_map
from scripts import inspect_r2ad_checkpoints as inspection


class MeanEncoder(nn.Module):
    def forward(self,x):
        mu=x.mean((2,3))
        return mu,torch.zeros_like(mu)


class MeanDecoder(nn.Module):
    def forward(self,x):
        return x[:,:,None,None].expand(-1,-1,128,128)


def test_optional_stitched_reconstruction_preserves_scores():
    frame=np.full((48,48,3),100,np.uint8)
    mask=np.ones((48,48),np.uint8)
    enc,dec=MeanEncoder(),MeanDecoder()
    baseline=infer_map(frame,mask,enc,dec,'cpu',patch=32,stride=16,sigma=0)
    result=infer_map(frame,mask,enc,dec,'cpu',patch=32,stride=16,sigma=0,return_reconstruction=True)
    assert np.array_equal(result['map'],baseline['map'])
    assert result['score']==baseline['score']
    assert result['reconstruction'].shape==frame.shape
    assert np.abs(result['reconstruction'].astype(float)-100).max()<=1


def test_inspection_uses_same_two_classes_for_each_checkpoint(tmp_path,monkeypatch):
    mask=tmp_path/'mask.png'
    cv2.imwrite(str(mask),np.full((48,48),255,np.uint8))
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'r2ad':{'safety_zone_mask':str(mask)}}))
    rows=[]
    for index,label in enumerate(('normal','anomalous')):
        source=tmp_path/f'frame_{index:06d}.png'
        cv2.imwrite(str(source),np.full((48,48,3),80+index*40,np.uint8))
        rows.append(dict(source=str(source),scenario=f'{index+1}_0',frame_id=index,label=label,split='test'))
    with (tmp_path/'test.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checkpoints=tmp_path/'models';checkpoints.mkdir()
    (checkpoints/'model_r2ad.pt').touch()
    (checkpoints/'checkpoints').mkdir()
    (checkpoints/'checkpoints'/'model_r2ad_epoch_0001.pt').touch()
    monkeypatch.setattr(inspection,'load_models',lambda *args:(MeanEncoder(),MeanDecoder(),{'epoch':1,'config':{'safety_area':'SafetyZone'}}))
    output=tmp_path/'inspection'
    monkeypatch.setattr(sys,'argv',['inspect','--config',str(config),'--split-dir',str(tmp_path),
        '--checkpoint-dir',str(checkpoints),'--output-dir',str(output),'--samples-per-class','1',
        '--patch-size','32','--stride','16','--sigma','0','--device','cpu'])
    inspection.main()
    with (output/'scores.csv').open() as stream:
        scores=list(csv.DictReader(stream))
    assert len(scores)==4
    for checkpoint in {row['checkpoint'] for row in scores}:
        subset=[row for row in scores if row['checkpoint']==checkpoint]
        assert {row['label'] for row in subset}=={'normal','anomalous'}
        assert {row['source'] for row in subset}=={row['source'] for row in rows}
    assert len(list(output.glob('*/*_comparison.jpg')))==4
    assert len(list(output.glob('*/*_reconstruction.png')))==4
    assert (output/'index.html').is_file()
    with (output/'class_summary.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==4
    assert json.loads((output/'report.json').read_text())['sample_count']==2
