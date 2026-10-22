"""Read virtual image splits without copying or linking source frames."""
import csv
from pathlib import Path


def read_frame_manifest(path):
    """Return source paths in CSV order; relative paths are relative to the CSV."""
    path = Path(path).expanduser().resolve()
    paths = []
    seen = set()
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'source' not in reader.fieldnames:
            raise ValueError(f'Frame manifest requires a source column: {path}')
        for line, row in enumerate(reader, 2):
            value = row.get('source', '')
            if not value or not value.strip():
                raise ValueError(f'Empty source at {path}:{line}')
            source = Path(value).expanduser()
            source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
            if not source.is_file():
                raise FileNotFoundError(f'Missing source frame at {path}:{line}: {source}')
            if source in seen:
                raise ValueError(f'Duplicate source frame at {path}:{line}: {source}')
            seen.add(source)
            paths.append(source)
    return paths
