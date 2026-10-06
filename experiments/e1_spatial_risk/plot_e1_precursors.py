import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_scenario(npz_path: Path, output_dir: Path, fps: float):
    d = np.load(npz_path, allow_pickle=True)

    scores = d["scores"].astype(np.float32)
    thresholds = d["thresholds"].astype(np.float32)
    areas = d["areas"]

    norm = scores / np.maximum(thresholds, 1e-12)
    t = np.arange(len(scores)) / fps

    scenario = npz_path.stem.replace("scenario_", "").replace("_maps", "")

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, area in enumerate(areas):
        ax.plot(
            t,
            norm[:, i],
            linewidth=1.7,
            label=str(area),
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.6,
        label="ADVIS threshold",
    )

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Normalized anomaly score", fontsize=13)
    ax.set_title(
        f"E1.0 — Temporal anomaly evidence — Scenario {scenario}",
        fontsize=15,
    )

    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, ncol=3)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    out = output_dir / f"scenario_{scenario}_precursors.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] {out}")

    # -------------------------------------------------
    # Print useful statistics
    # -------------------------------------------------
    print(f"\n=== Scenario {scenario} ===")

    for i, area in enumerate(areas):
        x = norm[:, i]
        anomalous = x > 1.0

        starts = np.where(
            anomalous & np.r_[True, ~anomalous[:-1]]
        )[0]

        print(
            f"{str(area):8s} "
            f"min={x.min():6.3f} "
            f"mean={x.mean():6.3f} "
            f"max={x.max():6.3f} "
            f"anom={100 * anomalous.mean():6.1f}% "
            f"episodes={len(starts):3d}"
        )


def create_summary(input_dir: Path, output_dir: Path, fps: float):
    files = sorted(input_dir.glob("scenario_*_maps.npz"))

    if not files:
        raise FileNotFoundError(
            f"No scenario_*_maps.npz files found in {input_dir}"
        )

    # One figure per area.
    first = np.load(files[0], allow_pickle=True)
    areas = list(first["areas"])

    for area_idx, area in enumerate(areas):

        fig, ax = plt.subplots(figsize=(16, 8))

        for npz_path in files:
            d = np.load(npz_path, allow_pickle=True)

            scores = d["scores"].astype(np.float32)
            thresholds = d["thresholds"].astype(np.float32)

            norm = scores / np.maximum(thresholds, 1e-12)
            t = np.arange(len(scores)) / fps

            scenario = (
                npz_path.stem
                .replace("scenario_", "")
                .replace("_maps", "")
            )

            ax.plot(
                t,
                norm[:, area_idx],
                linewidth=1.3,
                alpha=0.8,
                label=scenario,
            )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.8,
            label="ADVIS threshold",
        )

        ax.set_xlabel("Time (s)", fontsize=13)
        ax.set_ylabel("Normalized anomaly score", fontsize=13)

        ax.set_title(
            f"E1.0 — All scenarios — {area}",
            fontsize=15,
        )

        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.25)

        ax.legend(
            fontsize=9,
            ncol=4,
            loc="upper right",
        )

        fig.tight_layout()

        out = output_dir / f"all_scenarios_{area}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[save] {out}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="results/V6/e1_spatial_risk/exports",
    )

    parser.add_argument(
        "--output-dir",
        default="results/V6/e1_spatial_risk/precursor_plots",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(input_dir.glob("scenario_*_maps.npz"))

    print(f"[found] {len(files)} scenarios")

    for npz_path in files:
        plot_scenario(
            npz_path,
            output_dir,
            args.fps,
        )

    print("\nCreating all-scenario comparison plots...")

    create_summary(
        input_dir,
        output_dir,
        args.fps,
    )


if __name__ == "__main__":
    main()