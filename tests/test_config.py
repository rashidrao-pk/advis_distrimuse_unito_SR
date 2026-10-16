from argparse import Namespace
from pathlib import Path
import numpy as np
import cv2
import yaml
from r2ad.config import resolve


def test_config_resolution(tmp_path):
    root=tmp_path/'V6'; (root/'train').mkdir(parents=True); (root/'val').mkdir()
    (root/'test').mkdir(); (root/'normal_test').mkdir()
    mask=tmp_path/'mask.png';cv2.imwrite(str(mask),np.full((720,1280),255,np.uint8))
    cfg={'data':{'dataset_base':str(root),'mask_types':{'PLeft':str(mask)}},
         'models':{'latent_dims':64}, 'training':{'epochs':5,'batch_size':8,'num_workers':0},
         'results':{'output_dir':str(tmp_path/'results')},
         'r2ad':{'dataset_version':'V6','train_frames':'train','val_frames':'val',
                 'test_frames':'test','normal_test_frames':'normal_test'}}
    cfg_file=tmp_path/'cf.yaml';cfg_file.write_text(yaml.safe_dump(cfg))
    args=Namespace(config=str(cfg_file), safety_area='PLeft', train_dir=None, val_dir=None,
                   mask=None, output=None, latent_dims=None, epochs=None,batch_size=None,workers=None)
    args=resolve(args,'train')
    assert args.mask==str(mask)
    assert args.latent_dims==64 and args.epochs==5 and args.batch_size==8
    assert args.train_dir==str(root/'train')
    assert args.output.endswith('/V6/r2ad/PLeft/model_r2ad.pt')
