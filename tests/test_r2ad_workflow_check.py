import csv
import subprocess
import sys
from pathlib import Path

import yaml


def test_checker_valid_split_and_leakage(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data': {'dataset_base': str(tmp_path)}, 'r2ad': {}}))
    split_dir = tmp_path / 'split'
    split_dir.mkdir()
    sources = []
    for scenario in ('1_0', '2_0', '8_0'):
        folder = tmp_path / f'scenario_{scenario}'
        folder.mkdir()
        source = folder / 'frame_000000.png'
        source.write_bytes(b'placeholder')
        sources.append(source)
    def write(name, split, source, label):
        with (split_dir / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['source', 'split', 'label'])
            writer.writeheader()
            writer.writerow(dict(source=str(source), split=split, label=label))
    write('train', 'train', sources[0], 'normal')
    write('val', 'val', sources[1], 'normal')
    write('test', 'test', sources[2], 'normal')
    write('normal_test', 'test', sources[2], 'normal')
    script = Path(__file__).resolve().parents[1] / 'scripts/check_r2ad_workflow.py'
    command = [sys.executable, str(script), '--config', str(config),
               '--split-dir', str(split_dir), '--run-dir', str(tmp_path / 'run')]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert '[PASS] Scenario separation' in output
    assert '[PASS] Normal test subset' in output
    assert '[REVIEW] Test classes' in output
    assert '[PENDING] Checkpoint' in output
    write('val', 'val', sources[0], 'normal')
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    assert '[FAIL] Scenario separation' in output
