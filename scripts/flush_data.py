"""
flush_data.py
-------------
Delete trained VAE-GAN model checkpoints and calibrated threshold JSON files
for one safety area or all safety areas.

Examples
--------
# Dry run for one area
python scripts/flush_data.py \
    --safety_area RoboArm \
    --latent_dims 64 \
    --checkpoints scripts/results/models \
    --threshold_dir scripts/results/thresholds \
    --dry_run

# Actually delete for one area
python scripts/flush_data.py \
    --safety_area RoboArm \
    --latent_dims 64 \
    --checkpoints scripts/results/models \
    --threshold_dir scripts/results/thresholds

# Delete for all areas
python scripts/flush_data.py \
    --safety_area ALL \
    --latent_dims 64 \
    --checkpoints scripts/results/models \
    --threshold_dir scripts/results/thresholds
"""

import os
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path


ALL_SAFETY_AREAS = ["RoboArm", "ConvBelt", "PLeft", "PRight"]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_model_candidates(checkpoints_dir: Path, safety_area: str, latent_dims: int):
    """
    Expected primary file from your train.py / save_model convention:
        model_{safety_area}_{latent_dims}.pt

    Also removes common companion files if present.
    """
    suffix = f"{safety_area}_{latent_dims}"

    candidates = [
        checkpoints_dir / f"model_{suffix}.pt",
        checkpoints_dir / f"history_{suffix}.csv",
        checkpoints_dir / f"{suffix}_config.json",
        checkpoints_dir / f"log_{suffix}.txt",
    ]

    return candidates, suffix


def build_threshold_candidates(threshold_dir: Path, safety_area: str):
    """
    Expected threshold file from your inference convention:
        threshold_dir / safety_area / threshold_{safety_area}.json
    """
    area_dir = threshold_dir / safety_area
    candidates = [
        area_dir / f"threshold_{safety_area}.json",
        area_dir / f"calibration_{safety_area}.json",
        area_dir / f"stats_{safety_area}.json",
    ]
    return candidates, area_dir


def delete_path(path: Path, dry_run: bool = False, verbose: bool = True):
    if not path.exists():
        if verbose:
            print(f"[skip] Not found: {path}")
        return False

    if dry_run:
        print(f"[dry-run] Would remove: {path}")
        return True

    if path.is_dir():
        shutil.rmtree(path)
        print(f"[removed-dir] {path}")
    else:
        path.unlink()
        print(f"[removed-file] {path}")
    return True


def flush_one_area(
    safety_area: str,
    latent_dims: int,
    checkpoints_dir: Path,
    threshold_dir: Path,
    dry_run: bool = False,
    remove_empty_threshold_dir: bool = True,
):
    print("\n" + "=" * 100)
    print(f"[{_now()}] Flushing area: {safety_area}")
    print("=" * 100)

    removed_any = False

    model_candidates, suffix = build_model_candidates(checkpoints_dir, safety_area, latent_dims)
    print(f"[models] suffix={suffix}")
    for p in model_candidates:
        removed_any = delete_path(p, dry_run=dry_run) or removed_any

    threshold_candidates, area_threshold_dir = build_threshold_candidates(threshold_dir, safety_area)
    print(f"[thresholds] area_dir={area_threshold_dir}")
    for p in threshold_candidates:
        removed_any = delete_path(p, dry_run=dry_run) or removed_any

    if remove_empty_threshold_dir and area_threshold_dir.exists():
        try:
            if not any(area_threshold_dir.iterdir()):
                if dry_run:
                    print(f"[dry-run] Would remove empty dir: {area_threshold_dir}")
                else:
                    area_threshold_dir.rmdir()
                    print(f"[removed-empty-dir] {area_threshold_dir}")
                removed_any = True
        except Exception as e:
            print(f"[warn] Could not remove empty threshold dir {area_threshold_dir}: {e}")

    if not removed_any:
        print(f"[info] Nothing found to remove for {safety_area}")

    return removed_any


def write_summary_log(
    out_path: Path,
    payload: dict,
    dry_run: bool = False,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] Would write summary log: {out_path}")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved-log] {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Flush trained models and thresholds for CAD/ADVIS.")

    p.add_argument(
        "--safety_area",
        default="RoboArm",
        help="Safety area to flush. Use 'ALL' for all areas."
    )
    p.add_argument(
        "--latent_dims",
        type=int,
        default=64,
        help="Latent dimension used in saved model suffix."
    )
    p.add_argument(
        "--checkpoints",
        default="scripts/results/models",
        help="Directory containing saved model checkpoints."
    )
    p.add_argument(
        "--threshold_dir",
        default="scripts/results/thresholds",
        help="Directory containing saved threshold JSON files."
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be deleted."
    )
    p.add_argument(
        "--keep_threshold_dirs",
        action="store_true",
        help="Do not remove empty per-area threshold directories."
    )
    p.add_argument(
        "--save_log",
        action="store_true",
        help="Save a JSON log of the flush operation."
    )
    p.add_argument(
        "--log_path",
        default="scripts/results/flush_logs/flush_summary.json",
        help="Where to save the flush summary log."
    )

    return p.parse_args()


def main():
    args = parse_args()

    checkpoints_dir = Path(args.checkpoints).resolve()
    threshold_dir = Path(args.threshold_dir).resolve()

    print(f"[start] { _now() }")
    print(f"[checkpoints_dir] {checkpoints_dir}")
    print(f"[threshold_dir]  {threshold_dir}")
    print(f"[dry_run]        {args.dry_run}")

    if args.safety_area.upper() == "ALL":
        areas = ALL_SAFETY_AREAS
    else:
        areas = [args.safety_area]

    summary = {
        "timestamp": _now(),
        "dry_run": args.dry_run,
        "latent_dims": args.latent_dims,
        "checkpoints_dir": str(checkpoints_dir),
        "threshold_dir": str(threshold_dir),
        "areas": areas,
    }

    removed_flags = {}
    for area in areas:
        removed = flush_one_area(
            safety_area=area,
            latent_dims=args.latent_dims,
            checkpoints_dir=checkpoints_dir,
            threshold_dir=threshold_dir,
            dry_run=args.dry_run,
            remove_empty_threshold_dir=not args.keep_threshold_dirs,
        )
        removed_flags[area] = removed

    summary["removed_any"] = removed_flags

    if args.save_log:
        write_summary_log(Path(args.log_path).resolve(), summary, dry_run=args.dry_run)

    print("\n[done] Flush complete.")


if __name__ == "__main__":
    main()