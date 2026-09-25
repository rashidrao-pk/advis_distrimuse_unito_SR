#!/usr/bin/env python3
"""Build a per-scenario test dataset from safety-area annotation CSVs.

Output layout::

    test/<scenario>/<safety-area>/<normal|anomalous|verify>/<cropped-frame>
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cf_dataset_mac.yaml"
DEFAULT_ANNOTATIONS = (
    PROJECT_ROOT / "reports" / "safety_area_annotations" / "saved_annotation"
)
CLASS_NAMES = ("normal", "anomalous", "verify")
LABEL_ALIASES = {
    "normal": "normal",
    "anomalous": "anomalous",
    "anomaly": "anomalous",
    "verify": "verify",
    "intermediate": "verify",
}
REQUIRED_COLUMNS = {
    "scenario_id",
    "camera",
    "safety_area",
    "filename",
    "label",
    "processed_image_path",
}


@dataclass(frozen=True)
class TestFrame:
    scenario_id: str
    source_scenario_id: str
    scenario_description: str
    camera: str
    safety_area: str
    frame_id: str
    source_frame_id: str
    label: str
    note: str
    source: Path
    annotation_csv: Path


def safe_component(value: str, field: str) -> str:
    value = str(value).strip()
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def normalize_label(value: str) -> str | None:
    key = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key in {"", "unlabelled", "unlabeled"}:
        return None
    return LABEL_ALIASES.get(key)


def load_paths(config_path: Path) -> tuple[Path, Path]:
    with config_path.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    data = config.get("data") or {}
    dataset_base = data.get("dataset_base")
    if not dataset_base:
        raise ValueError(f"{config_path} does not define data.dataset_base")
    dataset_base = Path(dataset_base).expanduser()
    extracted_root = dataset_base / "extracted_frames"
    output_root = Path(data.get("testing") or dataset_base / "test").expanduser()
    return extracted_root, output_root


def annotation_files(annotation_dir: Path) -> list[Path]:
    return sorted(annotation_dir.glob("scenario_*_annotations.csv"))


def infer_scenario_id_from_annotation(path: Path, camera: str) -> str | None:
    match = re.fullmatch(
        rf"scenario_(.+)_{re.escape(camera)}_annotations\.csv", path.name
    )
    return match.group(1) if match else None


def resolve_source(
    row: dict[str, str], extracted_root: Path, source_scenario_id: str
) -> Path:
    annotated = Path((row.get("processed_image_path") or "").strip()).expanduser()
    if annotated.is_file():
        return annotated.resolve()

    return (
        extracted_root
        / source_scenario_id
        / row["camera"].strip()
        / "processed"
        / row["safety_area"].strip()
        / row["filename"].strip()
    ).resolve()


def read_annotations(
    files: Iterable[Path],
    extracted_root: Path,
    camera: str,
    scenarios: set[str] | None,
    safety_areas: set[str] | None,
    cumulative_scenario_id: str | None = None,
) -> tuple[list[TestFrame], dict[str, int]]:
    records: list[TestFrame] = []
    stats = {
        "rows": 0,
        "other_camera": 0,
        "unselected": 0,
        "unlabelled": 0,
        "unsupported_label": 0,
        "missing_source": 0,
        "duplicates": 0,
    }
    seen: dict[tuple[str, str, str, str], TestFrame] = {}

    for csv_path in files:
        inferred_scenario_id = infer_scenario_id_from_annotation(csv_path, camera)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    f"{csv_path} is missing column(s): "
                    + ", ".join(sorted(missing_columns))
                )

            for row in reader:
                stats["rows"] += 1
                row_camera = row["camera"].strip()
                if row_camera != camera:
                    stats["other_camera"] += 1
                    continue

                annotated_scenario_id = str(row["scenario_id"]).strip()
                is_unified = annotated_scenario_id.lower() in {
                    "unified", "cumulative", "combined"
                }
                scenario_id = (
                    cumulative_scenario_id
                    or (inferred_scenario_id if is_unified else annotated_scenario_id)
                )
                if not scenario_id:
                    raise ValueError(
                        f"Cannot determine cumulative scenario ID for {csv_path}. "
                        "Use --cumulative-scenario-id."
                    )
                scenario_id = safe_component(scenario_id, "scenario_id")
                source_scenario_id = safe_component(
                    row.get("source_scenario_id") or annotated_scenario_id,
                    "source_scenario_id",
                )
                safety_area = safe_component(row["safety_area"], "safety_area")
                if scenarios is not None and scenario_id not in scenarios:
                    stats["unselected"] += 1
                    continue
                if safety_areas is not None and safety_area not in safety_areas:
                    stats["unselected"] += 1
                    continue

                raw_label = str(row["label"] or "").strip()
                label = normalize_label(raw_label)
                if not raw_label or raw_label.lower() in {"unlabelled", "unlabeled"}:
                    stats["unlabelled"] += 1
                    continue
                if label is None or label not in CLASS_NAMES:
                    stats["unsupported_label"] += 1
                    continue

                source = resolve_source(row, extracted_root, source_scenario_id)
                if not source.is_file():
                    stats["missing_source"] += 1
                    continue

                filename = safe_component(row["filename"], "filename")
                frame_id = (row.get("frame_id") or Path(filename).stem).strip()
                record = TestFrame(
                    scenario_id=scenario_id,
                    source_scenario_id=source_scenario_id,
                    scenario_description=(row.get("scenario_description") or "").strip(),
                    camera=row_camera,
                    safety_area=safety_area,
                    frame_id=frame_id,
                    source_frame_id=(row.get("source_frame_id") or frame_id).strip(),
                    label=label,
                    note=(row.get("note") or "").strip(),
                    source=source,
                    annotation_csv=csv_path.resolve(),
                )
                key = (scenario_id, row_camera, safety_area, frame_id)
                previous = seen.get(key)
                if previous is not None:
                    stats["duplicates"] += 1
                    if previous.label != record.label or previous.source != record.source:
                        raise ValueError(
                            "Conflicting annotations for "
                            f"{scenario_id}/{row_camera}/{safety_area}/frame {frame_id}: "
                            f"{previous.label} versus {record.label}"
                        )
                    continue
                seen[key] = record
                records.append(record)

    records.sort(
        key=lambda item: (
            tuple(int(part) if part.isdigit() else sys.maxsize for part in item.scenario_id.split("_")),
            item.safety_area,
            int(item.frame_id) if item.frame_id.isdigit() else sys.maxsize,
            item.source.name,
        )
    )
    return records, stats


def transfer(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        destination.hardlink_to(source)
    else:
        destination.symlink_to(source)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "scenario_id",
        "source_scenario_id",
        "scenario_description",
        "camera",
        "safety_area",
        "frame_id",
        "source_frame_id",
        "label",
        "note",
        "source_path",
        "test_path",
        "annotation_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Create test/<scenario>/<safety-area>/<label>/ from annotation CSVs."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--annotation-csv", type=Path,
        help=(
            "Use one annotation CSV instead of scanning --annotations-dir. "
            "This is recommended for a cumulative unified annotation."
        ),
    )
    parser.add_argument(
        "--extracted-root", type=Path,
        help="Default: <config data.dataset_base>/extracted_frames.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: config data.testing, or <dataset_base>/test.",
    )
    parser.add_argument("--camera", default="back_view")
    parser.add_argument("--scenarios", nargs="+", help="Default: all annotated scenarios.")
    parser.add_argument(
        "--cumulative-scenario-id",
        help=(
            "Output scenario ID for a unified annotation. By default it is "
            "inferred from names such as scenario_8_16_back_view_annotations.csv."
        ),
    )
    parser.add_argument("--safety-areas", nargs="+", help="Default: all annotated areas.")
    parser.add_argument(
        "--mode", choices=("copy", "hardlink", "symlink"), default="copy",
        help="How cropped frames are placed in the test tree (default: copy).",
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Complete the dataset even if some referenced cropped frames are missing.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_extracted, config_output = load_paths(args.config.resolve())
    extracted_root = (args.extracted_root or config_extracted).expanduser().resolve()
    output_root = (args.output_dir or config_output).expanduser().resolve()
    annotation_dir = args.annotations_dir.expanduser().resolve()

    if args.annotation_csv:
        selected_csv = args.annotation_csv.expanduser().resolve()
        if not selected_csv.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {selected_csv}")
        files = [selected_csv]
    else:
        if not annotation_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
        files = annotation_files(annotation_dir)
        if not files:
            raise FileNotFoundError(f"No annotation CSV files found in {annotation_dir}")
    if not extracted_root.is_dir():
        raise FileNotFoundError(f"Extracted-frame directory not found: {extracted_root}")

    requested_scenarios = set(args.scenarios) if args.scenarios else None
    requested_areas = set(args.safety_areas) if args.safety_areas else None
    records, read_stats = read_annotations(
        files,
        extracted_root,
        args.camera,
        requested_scenarios,
        requested_areas,
        args.cumulative_scenario_id,
    )
    if read_stats["missing_source"] and not args.skip_missing:
        raise FileNotFoundError(
            f"{read_stats['missing_source']} annotated cropped frame(s) are missing. "
            "Use --skip-missing to omit them."
        )
    if not records:
        raise ValueError("No usable annotated frames matched the requested selection")

    scenarios_and_areas = sorted({
        (record.scenario_id, record.safety_area) for record in records
    })
    if not args.dry_run:
        for scenario_id, safety_area in scenarios_and_areas:
            for class_name in CLASS_NAMES:
                (output_root / scenario_id / safety_area / class_name).mkdir(
                    parents=True, exist_ok=True
                )

    iterable = records
    if args.progress:
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise RuntimeError("--progress requires tqdm") from exc
        iterable = tqdm(records, desc="Creating test dataset", unit="frame")

    written = 0
    existing = 0
    manifest_rows: list[dict[str, str]] = []
    claimed: dict[Path, Path] = {}
    class_counts = {name: 0 for name in CLASS_NAMES}
    area_counts: dict[tuple[str, str], int] = {}

    for record in iterable:
        destination = (
            output_root
            / record.scenario_id
            / record.safety_area
            / record.label
            / record.source.name
        )
        previous_source = claimed.get(destination)
        if previous_source is not None and previous_source != record.source:
            raise ValueError(
                f"Destination collision: {destination} maps to both "
                f"{previous_source} and {record.source}"
            )
        claimed[destination] = record.source

        class_counts[record.label] += 1
        area_key = (record.scenario_id, record.safety_area)
        area_counts[area_key] = area_counts.get(area_key, 0) + 1
        if destination.exists() or destination.is_symlink():
            existing += 1
        else:
            if not args.dry_run:
                transfer(record.source, destination, args.mode)
            written += 1

        manifest_rows.append({
            "scenario_id": record.scenario_id,
            "source_scenario_id": record.source_scenario_id,
            "scenario_description": record.scenario_description,
            "camera": record.camera,
            "safety_area": record.safety_area,
            "frame_id": record.frame_id,
            "source_frame_id": record.source_frame_id,
            "label": record.label,
            "note": record.note,
            "source_path": str(record.source),
            "test_path": str(destination),
            "annotation_csv": str(record.annotation_csv),
        })

    manifest_path = output_root / f"test_manifest_{record.scenario_id}.csv"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_path, manifest_rows)

    print(
        "Annotations: "
        + (str(files[0]) if args.annotation_csv else str(annotation_dir))
    )
    print(f"Extracted crops: {extracted_root}")
    print(f"Test output: {output_root}")
    print(f"Camera: {args.camera}")
    print(f"Mode: {args.mode}{' (dry run)' if args.dry_run else ''}")
    print(f"Annotation CSVs: {len(files)}")
    print(f"Selected frames: {len(records)}")
    print(f"Written: {written}; already existing: {existing}")
    print(
        "Classes: "
        + ", ".join(f"{name}={class_counts[name]}" for name in CLASS_NAMES)
    )
    print(f"Scenario/area groups: {len(area_counts)}")
    if read_stats["unlabelled"] or read_stats["unsupported_label"]:
        print(
            "Skipped labels: "
            f"unlabelled={read_stats['unlabelled']}, "
            f"unsupported={read_stats['unsupported_label']}"
        )
    if read_stats["missing_source"]:
        print(f"Missing sources skipped: {read_stats['missing_source']}")
    if args.dry_run:
        print("Dry run complete; no directories or files were created.")
    else:
        print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
