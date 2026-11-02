import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.mark.parametrize('mode',['random','grid'])
def test_patch_animation_is_model_free_and_keeps_source_size(tmp_path,mode):
    frame=tmp_path/'frame.png';mask=tmp_path/'mask.png'
    cv2.imwrite(str(frame),np.full((64,64,3),120,np.uint8))
    cv2.imwrite(str(mask),np.full((64,64),255,np.uint8))
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'r2ad':{'safety_zone_mask':str(mask)}}))
    output=tmp_path/f'{mode}.gif'
    script=Path(__file__).resolve().parents[1]/'scripts/animate_r2ad_patch_sampling.py'
    command=[sys.executable,str(script),'--config',str(config),'--image',str(frame),'--output',str(output),
             '--patch-size','32','--stride','32','--sampling',mode,'--num-patches','3','--width','720',
             '--aug-rotate','0','--aug-zoom-min','1','--aug-zoom-max','1.03']
    subprocess.run(command,check=True,capture_output=True,text=True)
    report=json.loads(output.with_suffix('.json').read_text())
    assert report['source_patch_size']==32 and report['model_input_size']==128
    assert all(row['box'][2:]==[32,32] for row in report['samples'])
    with Image.open(output) as gif:
        assert gif.n_frames==report['animation_frames']
        assert gif.width==720
    assert output.with_suffix('.png').is_file()
    if mode=='grid':
        assert report['total_patches']==4
        assert report['samples'][-1]['coverage']==1.
    else:
        assert report['total_patches']==3
        assert all(row['augmentation']['angle']==0 for row in report['samples'])
