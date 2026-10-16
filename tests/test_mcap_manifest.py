from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.index_rosbags_r2ad import category_for
from r2ad.mcap_data import decode_image, FIELDS, load_manifest
import numpy as np
import csv


def test_category():
    cfg={'scenario_categories':{'1-7':'normal','8-16':'anomalous'}}
    assert category_for('1_0',cfg)=='normal'
    assert category_for('3_0',cfg)=='normal'
    assert category_for('13_1',cfg)=='unexpected'


def test_manifest_splits(tmp_path):
    p=tmp_path/'manifest.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for scenario in ('1_0','3_0'):
            for i in range(10):
                w.writerow(dict(scenario=scenario,label='normal',split=('val' if i%5==4 else 'train'),bag='foo',topic='bar',log_time=i,occurrence=0,ordinal=i))
    assert len(load_manifest(p,'train'))==16
    assert len(load_manifest(p,'val'))==4
    assert {r['scenario'] for r in load_manifest(p,'val')}=={'1_0','3_0'}


def test_image_decode_rgb():
    class Msg:
        encoding='rgb8';height=2;width=2;step=6
        data=bytes([255,0,0]*4)
    result=decode_image(Msg(),'sensor_msgs/msg/Image')
    assert tuple(result[0,0])==(0,0,255)
