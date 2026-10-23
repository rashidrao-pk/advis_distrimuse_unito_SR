"""Join authoritative per-area annotations to original extracted frame IDs."""
import csv
import re
from pathlib import Path

AREAS = ('PLeft', 'PRight', 'RoboArm', 'ConvBelt')
ANNOTATION_FIELDS = ['scenario', 'camera', 'frame_id', *[f'label_{a}' for a in AREAS]]


def load_area_annotations(path, camera):
    result = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'source_scenario_id', 'source_frame_id', 'camera', 'safety_area', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Combined annotations require columns: {sorted(required)}')
        for row in reader:
            if row['camera'].strip() != camera:
                continue
            area = row['safety_area'].strip()
            if area not in AREAS:
                continue
            key = (row['source_scenario_id'].strip(), int(row['source_frame_id']))
            label = row['label'].strip().lower()
            label = label if label in ('normal', 'anomalous') else 'unknown'
            labels = result.setdefault(key, {})
            if area in labels and labels[area] != label:
                raise ValueError(f'Conflicting annotations for {key}, {area}')
            labels[area] = label
    if not result:
        raise ValueError(f'No annotations for camera {camera}: {path}')
    return result


def annotate_manifest(rows, annotations, camera, unexpected_root):
    """Scenario categories define held-out recordings, never frame ground truth."""
    for row in rows:
        path = Path(row['source'])
        is_held_out = path.is_relative_to(unexpected_root)
        root = unexpected_root if is_held_out else None
        scenario = (path.relative_to(root).parts[0] if root else path.parent.name).removeprefix('scenario_')
        match = re.fullmatch(r'frame_(\d+)', path.stem)
        if not match:
            raise ValueError(f'Cannot match original frame index: {path}; expected frame_000000.png')
        frame_id = int(match.group(1))
        labels = annotations.get((scenario, frame_id), {}) if is_held_out else dict.fromkeys(AREAS, 'normal')
        labels = {area: labels.get(area, 'unknown') for area in AREAS}
        label = ('anomalous' if 'anomalous' in labels.values() else
                 'normal' if all(v == 'normal' for v in labels.values()) else 'unknown')
        row.update(scenario=scenario, camera=camera, frame_id=frame_id, label=label,
                   **{f'label_{area}': value for area, value in labels.items()})
