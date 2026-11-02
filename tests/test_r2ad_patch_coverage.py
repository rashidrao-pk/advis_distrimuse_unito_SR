import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.estimate_r2ad_patch_coverage import coverage_counts,grid_estimate


def test_exact_grid_coverage_and_overlap():
    mask=np.ones((64,64),np.uint8)
    row,rectangles,covered=grid_estimate(mask,{'area':mask},16,16,.85,.99)
    assert row['patch_count']==16 and row['optimistic_area_lower_bound']==16
    assert row['safety_zone_coverage']==1 and row['area_coverage']==1
    assert row['mean_contributions_per_safety_pixel']==1
    assert covered.all()
    row,rectangles,covered=grid_estimate(mask,{},16,8,.85,.99)
    assert row['patch_count']==49 and row['safety_zone_coverage']==1
    expected=np.zeros(mask.shape,np.int32)
    for x,y,w,h in rectangles:expected[y:y+h,x:x+w]+=1
    assert np.array_equal(coverage_counts(rectangles,mask.shape),expected)
    assert row['mean_contributions_per_safety_pixel']>1
    sparse=np.zeros((32,32),np.uint8);sparse[0,0]=1
    row,rectangles,covered=grid_estimate(sparse,{},16,16,1,.99)
    assert row['patch_count']==0 and row['safety_zone_coverage']==0


def test_coverage_estimator_outputs_and_random_budgets(tmp_path):
    mask=tmp_path/'mask.png'
    cv2.imwrite(str(mask),np.full((64,64),255,np.uint8))
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'data':{'dataset_base':str(tmp_path)},'r2ad':{'safety_zone_mask':str(mask)}}))
    output=tmp_path/'coverage'
    script=Path(__file__).resolve().parents[1]/'scripts/estimate_r2ad_patch_coverage.py'
    command=[sys.executable,str(script),'--config',str(config),'--output-dir',str(output),
             '--patch-sizes','16','32','--stride-ratios','1','.5','--random-budgets','2','4','--trials','2']
    subprocess.run(command,check=True,capture_output=True,text=True)
    report=json.loads((output/'report.json').read_text())
    assert len(report['grids'])==4
    assert len(report['random_sampling'])==4
    assert (output/'index.html').is_file()
    assert len(list(output.glob('p*_s*.jpg')))==4
    for patch in (16,32):
        selected=[r for r in report['random_sampling'] if r['patch_size']==patch]
        assert selected[0]['mean_safety_coverage']<=selected[1]['mean_safety_coverage']
    with (output/'grid_coverage.csv').open() as stream:
        assert len(list(csv.DictReader(stream)))==4
