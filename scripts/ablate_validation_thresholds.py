#!/usr/bin/env python3
"""Ablate anomaly-score parameters and thresholds on normal validation data."""

import argparse
import csv
from html import escape
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import calibrate_threshold as calibration  # noqa: E402


def comma_values(value, converter):
    try:
        return [converter(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normal-only ablation of anomaly scores and threshold strategies."
    )
    parser.add_argument("--config", type=Path,
                        default=Path("configs/cf_dataset_epito.yaml"))
    parser.add_argument("--safety_area", default="PLeft")
    parser.add_argument("--dataset_version", default="V6")
    parser.add_argument("--dataset_type", default=None)
    parser.add_argument("--latent_dims", type=int)
    parser.add_argument("--exp_type", default="E3")
    parser.add_argument("--mask_image_name", type=int, default=3015)
    parser.add_argument("--checkpoints", help="Overrides models.checkpoints in config.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--offsets", default="0,1,2,3")
    parser.add_argument("--sigmas", default="0,0.5,1.0,1.5")
    parser.add_argument("--quantiles", default="0.95,0.99,0.999,1.0")
    parser.add_argument("--threshold_percentiles", default="95,99,99.5,99.9")
    parser.add_argument("--threshold_n_sigmas", default="2,3,4")
    parser.add_argument("--target_normal_fpr", type=float, default=0.01)
    parser.add_argument("--max_images", type=int,
                        help="Optional evenly sampled validation-image limit.")
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def load_configuration(args):
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    data, models = config.get("data", {}) or {}, config.get("models", {}) or {}
    training_value = data.get("training") or data.get("train")
    if not training_value and data.get("dataset_base"):
        training_value = str(Path(data["dataset_base"]) / "train")
    if not training_value:
        raise ValueError("Config needs data.training, data.train, or data.dataset_base")
    training = Path(training_value).expanduser()
    if not training.is_absolute(): training = config_path.parent / training
    checkpoint_value = args.checkpoints or models.get("checkpoints")
    if not checkpoint_value:
        raise ValueError("Config needs models.checkpoints or pass --checkpoints")
    checkpoints = Path(checkpoint_value).expanduser()
    if not checkpoints.is_absolute(): checkpoints = config_path.parent.parent / checkpoints
    args.checkpoints = str(checkpoints.resolve())
    args.latent_dims = args.latent_dims or int(models.get("latent_dims", 64))
    args.verbose_level = 1
    args.save_path_type = "cloud"
    return training.resolve()


def find_split(area_dir, area):
    candidates = [
        area_dir / f"split_scenario_4train_1val_{area}.json",
        area_dir / f"split_4train_1val_{area}.json",
        area_dir / f"split_frame_4train_1val_{area}.json",
    ]
    for path in candidates:
        if path.is_file(): return path
    raise FileNotFoundError("No validation split found. Checked:\n" +
                            "\n".join(str(path) for path in candidates))


def load_validation(area_dir, split_path, args):
    base = datasets.ImageFolder(str(area_dir), transform=calibration._val_transform())
    with split_path.open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    indices = list(split["val_indices"])
    if args.max_images and len(indices) > args.max_images:
        selected = np.linspace(0, len(indices) - 1, args.max_images, dtype=int)
        indices = [indices[index] for index in selected]
    loader = DataLoader(Subset(base, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, drop_last=False)
    return loader, [base.samples[index][0] for index in indices]


def make_method_specs(args):
    specs = [
        {"method": "MSE", "family": "baseline", "offset": math.nan,
         "sigma": math.nan, "quantile": math.nan},
        {"method": "MAE", "family": "baseline", "offset": math.nan,
         "sigma": math.nan, "quantile": math.nan},
    ]
    for offset in comma_values(args.offsets, int):
        if offset < 0: raise ValueError("Offsets must be non-negative")
        for sigma in comma_values(args.sigmas, float):
            if sigma < 0: raise ValueError("Sigmas must be non-negative")
            for quantile in comma_values(args.quantiles, float):
                if not 0 < quantile <= 1: raise ValueError("Quantiles must be in (0, 1]")
                specs.append({
                    "method": f"TAAS_off{offset}_sig{sigma:g}_q{quantile:g}",
                    "family": "TAAS", "offset": offset,
                    "sigma": sigma, "quantile": quantile,
                })
    return specs


def compute_scores(loader, Enc, Dec, specs, device):
    collected = {spec["method"]: [] for spec in specs}
    taas_specs = [spec for spec in specs if spec["family"] == "TAAS"]
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Reconstructing and scoring", leave=False):
            reconstructed = calibration.reconstruct(Enc, Dec, images, device)
            for batch_index in range(len(images)):
                original = calibration.tensor_to_hwc(images[batch_index])
                reconstruction = calibration.tensor_to_hwc(reconstructed[batch_index])
                residual = np.abs(original - reconstruction)
                collected["MSE"].append(float(np.mean(residual ** 2)))
                collected["MAE"].append(float(np.mean(residual)))
                distance_cache = {}
                smooth_cache = {}
                for spec in taas_specs:
                    offset, sigma = spec["offset"], spec["sigma"]
                    if offset not in distance_cache:
                        distance_cache[offset] = calibration._distance_offset_np(
                            original, reconstruction, offset
                        )
                    cache_key = (offset, sigma)
                    if cache_key not in smooth_cache:
                        distance = distance_cache[offset]
                        smooth_cache[cache_key] = (
                            calibration.gaussian_filter(distance, sigma=sigma)
                            if sigma > 0 else distance
                        )
                    collected[spec["method"]].append(float(np.quantile(
                        smooth_cache[cache_key], spec["quantile"]
                    )))
    return {key: np.asarray(values, dtype=float) for key, values in collected.items()}


def threshold_specs(args):
    result = [{"strategy": "max", "parameter": math.nan}]
    result.extend({"strategy": "percentile", "parameter": value}
                  for value in comma_values(args.threshold_percentiles, float))
    result.extend({"strategy": "mean_std", "parameter": value}
                  for value in comma_values(args.threshold_n_sigmas, float))
    return result


def threshold_for(scores, spec):
    if spec["strategy"] == "max": return float(scores.max())
    if spec["strategy"] == "percentile":
        return float(np.percentile(scores, spec["parameter"]))
    return float(scores.mean() + spec["parameter"] * scores.std())


def analyze(specs, scores_by_method, args):
    rows = []
    for method_spec in specs:
        scores = scores_by_method[method_spec["method"]]
        calibration_scores, evaluation_scores = scores[::2], scores[1::2]
        for threshold_spec in threshold_specs(args):
            threshold = threshold_for(calibration_scores, threshold_spec)
            calibration_fpr = float(np.mean(calibration_scores > threshold))
            evaluation_fpr = float(np.mean(evaluation_scores > threshold))
            mean = float(scores.mean())
            row = {
                **method_spec,
                "threshold_strategy": threshold_spec["strategy"],
                "threshold_parameter": threshold_spec["parameter"],
                "threshold": threshold,
                "threshold_over_mean": threshold / mean if mean else math.nan,
                "score_mean": mean, "score_std": float(scores.std()),
                "score_cv": float(scores.std() / mean) if mean else math.nan,
                "score_p95": float(np.percentile(scores, 95)),
                "score_p99": float(np.percentile(scores, 99)),
                "calibration_normal_fpr": calibration_fpr,
                "evaluation_normal_fpr": evaluation_fpr,
                "normal_fpr_gap": abs(evaluation_fpr - calibration_fpr),
                "target_fpr_error": abs(evaluation_fpr - args.target_normal_fpr),
                "n_calibration": len(calibration_scores),
                "n_evaluation": len(evaluation_scores),
            }
            row["normal_stability_rank_score"] = (
                row["target_fpr_error"] + row["normal_fpr_gap"]
            )
            rows.append(row)
    return sorted(rows, key=lambda row: row["normal_stability_rank_score"])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_score_csv(path, image_paths, scores):
    methods = list(scores)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", *methods])
        for index, image_path in enumerate(image_paths):
            writer.writerow([image_path, *[scores[method][index] for method in methods]])


def write_html(path, rows, area, target_fpr):
    top = rows[:min(80, len(rows))]
    figure = make_subplots(rows=2, cols=2, subplot_titles=(
        "Top configurations: evaluation normal FPR",
        "Threshold scale versus evaluation FPR",
        "Calibration FPR versus evaluation FPR",
        "Normal-score variability versus stability"))
    labels = [f'{row["method"]} | {row["threshold_strategy"]} {row["threshold_parameter"]:g}'
              if math.isfinite(row["threshold_parameter"])
              else f'{row["method"]} | {row["threshold_strategy"]}' for row in top]
    custom = [[row["method"], row["threshold_strategy"], row["threshold_parameter"],
               row["threshold"], row["evaluation_normal_fpr"], row["normal_fpr_gap"]]
              for row in rows]
    hover = ("%{customdata[0]}<br>Threshold: %{customdata[1]} %{customdata[2]}"
             "<br>τ=%{customdata[3]:.6f}<br>Eval normal FPR=%{customdata[4]:.3%}"
             "<br>FPR gap=%{customdata[5]:.3%}<extra></extra>")
    figure.add_trace(go.Bar(x=labels, y=[row["evaluation_normal_fpr"] for row in top],
                            marker_color=[row["normal_stability_rank_score"] for row in top],
                            hovertext=labels, name="Normal FPR"), row=1, col=1)
    figure.add_hline(y=target_fpr, line_dash="dash", line_color="red", row=1, col=1)
    figure.add_trace(go.Scatter(x=[row["threshold_over_mean"] for row in rows],
                                y=[row["evaluation_normal_fpr"] for row in rows],
                                mode="markers", marker={"color": [row["score_cv"] for row in rows],
                                "colorscale": "Viridis", "showscale": True, "colorbar": {"title": "Score CV"}},
                                customdata=custom, hovertemplate=hover), row=1, col=2)
    figure.add_trace(go.Scatter(x=[row["calibration_normal_fpr"] for row in rows],
                                y=[row["evaluation_normal_fpr"] for row in rows], mode="markers",
                                customdata=custom, hovertemplate=hover), row=2, col=1)
    figure.add_trace(go.Scatter(x=[row["score_cv"] for row in rows],
                                y=[row["normal_stability_rank_score"] for row in rows], mode="markers",
                                customdata=custom, hovertemplate=hover), row=2, col=2)
    figure.update_xaxes(tickangle=65, row=1, col=1)
    figure.update_xaxes(title_text="Threshold / mean score", row=1, col=2)
    figure.update_yaxes(title_text="Evaluation normal FPR", row=1, col=2)
    figure.update_xaxes(title_text="Calibration normal FPR", row=2, col=1)
    figure.update_yaxes(title_text="Evaluation normal FPR", row=2, col=1)
    figure.update_xaxes(title_text="Score coefficient of variation", row=2, col=2)
    figure.update_yaxes(title_text="Normal stability rank score (lower is better)", row=2, col=2)
    figure.update_layout(height=1100, template="plotly_white", showlegend=False,
                         title=f"Normal-validation threshold ablation — {escape(area)}")
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row[key]))}</td>" for key in
        ("method", "threshold_strategy", "threshold_parameter", "threshold",
         "evaluation_normal_fpr", "normal_fpr_gap", "score_cv")) + "</tr>"
        for row in rows[:100]
    )
    plot = figure.to_html(include_plotlyjs=True, full_html=False)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Threshold ablation</title>
<style>body{{font-family:system-ui;background:#f4f6f8;margin:0}}main{{max-width:1800px;margin:auto;padding:22px}}section{{background:white;padding:18px;border-radius:10px;margin:15px 0}}.warning{{border-left:5px solid #d97706}}.table{{max-height:650px;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border:1px solid #d7dde5;text-align:left}}th{{position:sticky;top:0;background:#e8eef5}}</style></head><body><main>
<h1>Normal-validation threshold ablation — {escape(area)}</h1>
<section class="warning"><b>Interpretation limit:</b> normal-only validation can compare threshold stability and false alarms, but cannot determine which score best detects anomalies. Confirm the shortlisted configurations on labeled anomalous frames.</section>
<section><h2>How the tolerance parameters work</h2><ul><li><b>offset:</b> minimum color distance within a ±offset spatial neighborhood. Larger values tolerate reconstruction shifts but can hide small anomalies.</li><li><b>sigma:</b> Gaussian smoothing of the distance map. Larger values suppress isolated noise but blur small defects.</li><li><b>quantile:</b> selected upper residual percentile. 1.0 uses the worst pixel; lower values ignore a small fraction of extreme pixels.</li></ul></section>
<section>{plot}</section><section class="table"><h2>Top 100 by normal calibration stability</h2><table><thead><tr><th>Score</th><th>Threshold strategy</th><th>Parameter</th><th>Threshold</th><th>Evaluation normal FPR</th><th>FPR gap</th><th>Score CV</th></tr></thead><tbody>{table_rows}</tbody></table></section>
</main></body></html>""", encoding="utf-8")


def main():
    args = parse_args()
    training = load_configuration(args)
    area_dir = training / args.safety_area
    if not area_dir.is_dir(): raise FileNotFoundError(f"Training area not found: {area_dir}")
    split_path = find_split(area_dir, args.safety_area)
    loader, image_paths = load_validation(area_dir, split_path, args)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {device}\n[split] {split_path}\n[validation images] {len(image_paths)}")
    params, paths = calibration._setup_params_paths(args.safety_area, args)
    paths.train_dir_processed_subgroup = str(area_dir)
    Enc, Dec, suffix, epochs = calibration.load_model_for_area(
        args.safety_area, params, paths, args, device
    )
    specs = make_method_specs(args)
    print(f"[ablation] {len(specs)} anomaly-score configurations")
    scores = compute_scores(loader, Enc, Dec, specs, device)
    rows = analyze(specs, scores, args)
    output = (args.output_dir or Path("results") / args.dataset_version /
              "threshold_ablation" / args.safety_area).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "threshold_ablation.csv", rows)
    write_score_csv(output / "validation_scores_all_methods.csv", image_paths, scores)
    write_html(output / "threshold_ablation.html", rows, args.safety_area,
               args.target_normal_fpr)
    metadata = {"safety_area": args.safety_area, "checkpoint_suffix": suffix,
                "epochs": epochs, "split": str(split_path), "images": len(image_paths),
                "score_configurations": len(specs), "threshold_results": len(rows),
                "best_normal_stability_only": rows[0]}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[save] {output / 'threshold_ablation.html'}")
    print(f"[save] {output / 'threshold_ablation.csv'}")
    print("[note] Select the final score only after testing labeled anomalies.")


if __name__ == "__main__": main()
