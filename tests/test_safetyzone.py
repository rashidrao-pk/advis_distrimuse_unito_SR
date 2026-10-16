from pathlib import Path
import cv2
import numpy as np
import pytest
from r2ad.safetyzone import union_masks
from scripts.preprocess_r2ad import split_normal

def test_union_binary(tmp_path):
    sources={}
    for i,name in enumerate(('PLeft','PRight','RoboArm','ConvBelt')):
        image=np.zeros((64,64),np.uint8)
        image[i*10:i*10+20,i*10:i*10+20]=255
        p=tmp_path/f'{name}.png'
        cv2.imwrite(str(p),image)
        sources[name]=p
    merged,components=union_masks(sources)
    assert merged.dtype==np.uint8
    assert np.array_equal(merged,np.maximum.reduce(list(components.values())))
    assert merged.max()==1

def test_mask_size_mismatch(tmp_path):
    a,b=tmp_path/'a.png',tmp_path/'b.png'
    cv2.imwrite(str(a),np.ones((10,10),np.uint8)*255)
    cv2.imwrite(str(b),np.ones((9,10),np.uint8)*255)
    with pytest.raises(ValueError,match='dimensions'):
        union_masks(dict(a=a,b=b))

def test_group_no_leakage(tmp_path):
    paths=[]
    for g in range(5):
        d=tmp_path/f'scenario{g}';d.mkdir()
        for f in range(3):
            p=d/f'{f}.png';p.touch();paths.append(p)
    splits,groups=split_normal(paths,tmp_path,.2,.2,42,'grouped')
    assert len(splits['train'])==9
    assert len(splits['val'])==3
    assert len(splits['test'])==3
    assert len({str(p) for x in splits.values() for p in x})==15
