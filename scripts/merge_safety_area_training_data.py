#!/usr/bin/env python3
"""Merge per-scenario safety-area crops into a training dataset hierarchy."""

import argparse
from pathlib import Path
import shutil
import sys

import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_config(path):
    with path.expanduser().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    dataset_base = config.get("data", {}).get("dataset_base")
    if not dataset_base:
        raise ValueError("Config does not define data.dataset_base")
    return config, Path(dataset_base).expanduser()


def scenario_sort_key(path):
    try:
        return tuple(int(part) for part in path.name.split("_"))
    except ValueError:
        return (sys.maxsize, path.name)


def discover_scenarios(extracted_root, camera, requested):
    available = {
        path.name: path
        for path in extracted_root.iterdir()
        if path.is_dir() and (path / camera / "processed").is_dir()
    }
    if requested:
        missing = [scenario for scenario in requested if scenario not in available]
        if missing:
            raise ValueError(
                f"Scenario(s) missing {camera}/processed: {', '.join(missing)}"
            )
        return [available[scenario] for scenario in requested]
    return sorted(available.values(), key=scenario_sort_key)


def discover_areas(config, scenarios, camera, requested):
    configured = list((config.get("data", {}).get("mask_types") or {}).keys())
    existing = {
        area_dir.name
        for scenario in scenarios
        for area_dir in (scenario / camera / "processed").iterdir()
        if area_dir.is_dir()
    }
    areas = requested or [area for area in configured if area in existing]
    if not areas:
        areas = sorted(existing)
    unknown = [area for area in areas if area not in existing]
    if unknown:
        raise ValueError(
            f"Safety area(s) not found in extracted data: {', '.join(unknown)}"
        )
    return areas


def image_files(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def destination_name(source, scenario_id, area):
    """Preserve scenario-aware names, adding metadata only when absent."""
    name = source.name
    if scenario_id in name and area in name:
        return name
    return f"s-{scenario_id}_s-{area}_{name}"


def transfer(source, destination, mode):
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        destination.hardlink_to(source)
    else:
        destination.symlink_to(source.resolve())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge extracted safety-area images into training/<area>/<class>."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/cf_dataset_mac.yaml")
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        help="Default: <config data.dataset_base>/extracted_frames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <config data.dataset_base>/training",
    )
    parser.add_argument("--camera", default="back_view")
    parser.add_argument("--scenarios", nargs="+", help="Default: all scenarios.")
    parser.add_argument("--safety-areas", nargs="+", help="Default: all configured areas.")
    parser.add_argument("--class-label", default="normal")
    parser.add_argument(
        "--mode", choices=["copy", "hardlink", "symlink"], default="copy"
    )
    parser.add_argument(
        "--max-files-per-area",
        type=int,
        help="Optional total limit per area, useful for a small test.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_files_per_area is not None and args.max_files_per_area < 1:
        parser.error("--max-files-per-area must be at least 1")
    return args


def main():
    args = parse_args()
    config, dataset_base = load_config(args.config)
    extracted_root = (
        args.extracted_root.expanduser()
        if args.extracted_root
        else dataset_base / "extracted_frames"
    ).resolve()
    output_root = (
        args.output_dir.expanduser()
        if args.output_dir
        else dataset_base / "training"
    ).resolve()
    if not extracted_root.is_dir():
        raise FileNotFoundError(f"Extracted-frames root not found: {extracted_root}")

    scenarios = discover_scenarios(extracted_root, args.camera, args.scenarios)
    if not scenarios:
        raise ValueError(f"No scenarios found under {extracted_root}")
    areas = discover_areas(config, scenarios, args.camera, args.safety_areas)

    print(f"Extracted root: {extracted_root}")
    print(f"Training root: {output_root}")
    print(f"Camera: {args.camera}")
    print(f"Scenarios ({len(scenarios)}): {', '.join(path.name for path in scenarios)}")
    print(f"Safety areas: {', '.join(areas)}")
    print(f"Transfer mode: {args.mode}{' (dry run)' if args.dry_run else ''}")

    counts = {
        area: {"found": 0, "written": 0, "existing": 0, "collisions": 0}
        for area in areas
    }
    claimed_destinations = set()
    for scenario in scenarios:
        for area in areas:
            source_dir = scenario / args.camera / "processed" / area
            if not source_dir.is_dir():
                print(f"[WARNING] Missing area folder: {source_dir}")
                continue
            destination_dir = output_root / area / args.class_label
            if not args.dry_run:
                destination_dir.mkdir(parents=True, exist_ok=True)

            for source in image_files(source_dir):
                if (
                    args.max_files_per_area is not None
                    and counts[area]["written"] + counts[area]["existing"]
                    >= args.max_files_per_area
                ):
                    break
                counts[area]["found"] += 1
                destination = destination_dir / destination_name(
                    source, scenario.name, area
                )
                destination_key = str(destination)
                if destination.exists() or destination.is_symlink():
                    counts[area]["existing"] += 1
                    continue
                if destination_key in claimed_destinations:
                    counts[area]["collisions"] += 1
                    print(f"[WARNING] Destination collision: {destination}")
                    continue
                claimed_destinations.add(destination_key)
                if not args.dry_run:
                    transfer(source, destination, args.mode)
                counts[area]["written"] += 1

    print("\nSummary")
    print("Area             Found   Written   Existing   Collisions")
    for area in areas:
        item = counts[area]
        print(
            f"{area:<16} {item['found']:>6} {item['written']:>9} "
            f"{item['existing']:>10} {item['collisions']:>12}"
        )
    if args.dry_run:
        print("\nDry run complete; no directories or files were created.")


if __name__ == "__main__":
    main()
