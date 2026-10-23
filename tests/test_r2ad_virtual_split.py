import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from r2ad.core import images
from r2ad.dataset import RectangleDataset
from r2ad.frame_manifest import read_frame_manifest


@pytest.mark.parametrize('annotated', [False, True])
def test_manifest_split_end_to_end(tmp_path, annotated):
    frame = np.full((16, 16, 3), 128, dtype=np.uint8)
    for label, scenarios in [('normal', range(5)), ('unexpected', range(1))]:
        for scenario in scenarios:
            folder = tmp_path / label / f'scenario_{scenario}'
            folder.mkdir(parents=True)
            for index in range(2):
                assert cv2.imwrite(str(folder / f'frame_{index}.png'), frame)
    mask = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask), np.full((16, 16), 255, dtype=np.uint8))
    output = tmp_path / 'splits'
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'data': {'dataset_base': str(tmp_path), 'mask_types': {
            area: str(mask) for area in ('PLeft', 'PRight', 'RoboArm', 'ConvBelt')}},
        'r2ad': {'normal_source': str(tmp_path / 'normal'),
                 'unexpected_source': str(tmp_path / 'unexpected'),
                 'preprocessed_dir': str(output)},
    }))
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo / 'scripts/preprocess_r2ad.py'), '--config', str(config)]
    if annotated:
        annotation = tmp_path / 'annotations.csv'
        with annotation.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source_scenario_id', 'source_frame_id', 'camera', 'safety_area', 'label'])
            writer.writeheader()
            for index in range(2):
                for area in ('PLeft', 'PRight', 'RoboArm', 'ConvBelt'):
                    writer.writerow(dict(source_scenario_id='0', source_frame_id=index,
                                         camera='back_view', safety_area=area,
                                         label='Anomalous' if index == 1 and area == 'PLeft' else 'Normal'))
        command += ['--annotations', str(annotation)]
    subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    assert {p.name for p in output.iterdir()} == {
        'train.csv', 'val.csv', 'test.csv', 'normal_test.csv', 'manifest.csv', 'metadata.json'} | ({'unknown_test.csv'} if annotated else set())
    train, val, test = [set(images(output / f'{name}.csv')) for name in ('train', 'val', 'test')]
    assert len(train) == 6 and len(val) == 2 and len(test) == 4
    assert not (train & val or train & test or val & test)
    assert {p.parent for p in train}.isdisjoint(p.parent for p in val | test)
    assert len(images(output / 'normal_test.csv')) == (3 if annotated else 2)
    assert all(p.parent.parent.name == 'normal' for p in train | val)
    with (output / 'manifest.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and all(row['processed'] == '' for row in rows)
    if annotated:
        held_out = [row for row in rows if Path(row['source']).is_relative_to(tmp_path / 'unexpected')]
        assert [row['label'] for row in held_out] == ['normal', 'anomalous']
        assert all(row['split'] == 'test' for row in held_out)
        assert held_out[1]['label_PLeft'] == 'anomalous'
    dataset = RectangleDataset(output / 'train.csv', mask, patch=8,
                               samples_per_frame=1, augment=False)
    # Existing network input conversion resizes sampled patches to 128x128.
    assert tuple(dataset[0].shape) == (3, 128, 128)


def test_manifest_relative_missing_duplicate_and_invalid(tmp_path):
    source = tmp_path / 'frame.png'
    source.write_bytes(b'placeholder')
    manifest = tmp_path / 'train.csv'
    manifest.write_text('source\nframe.png\n')
    assert read_frame_manifest(manifest) == [source]
    manifest.write_text('source\nframe.png\nframe.png\n')
    with pytest.raises(ValueError, match='Duplicate'):
        read_frame_manifest(manifest)
    source.unlink()
    with pytest.raises(FileNotFoundError, match='Missing source'):
        read_frame_manifest(manifest)
    manifest.write_text('bag,split\nx,train\n')
    with pytest.raises(ValueError, match='source column'):
        read_frame_manifest(manifest)


def test_annotation_source_ids_unknown_and_conflict(tmp_path):
    from r2ad.annotations import load_area_annotations, annotate_manifest, AREAS
    annotation = tmp_path / 'combined.csv'
    header = 'scenario_id,frame_id,source_scenario_id,source_frame_id,camera,safety_area,label\n'
    annotation.write_text(header + ''.join(
        f'unified,999,8_0,17,back_view,{area},Normal\n' for area in AREAS))
    labels = load_area_annotations(annotation, 'back_view')
    root = tmp_path / 'unexpected'
    rows = [dict(source=str(root / 'scenario_8_0' / f'frame_{index:06d}.png'), split='test') for index in (17, 18)]
    annotate_manifest(rows, labels, 'back_view', root)
    assert [row['label'] for row in rows] == ['normal', 'unknown']
    assert rows[0]['frame_id'] == 17
    labels[('8_0', 18)] = {'PLeft': 'anomalous', 'PRight': 'unknown'}
    annotate_manifest(rows, labels, 'back_view', root)
    assert rows[1]['label'] == 'anomalous'
    with annotation.open('a') as stream:
        stream.write('unified,999,8_0,17,back_view,PLeft,Anomalous\n')
    with pytest.raises(ValueError, match='Conflicting'):
        load_area_annotations(annotation, 'back_view')
