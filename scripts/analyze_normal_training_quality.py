#!/usr/bin/env python3
"""Audit one-class normal safety-area images for quality and variability."""

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image
import yaml


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SCENARIO_PATTERN = re.compile(r"(?:^|[_-])s-(\d+_\d+)(?:[_-]|$)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit one-class normal training images for quality and variability."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/cf_dataset_mac.yaml"))
    parser.add_argument("--safety-area", required=True)
    parser.add_argument("--training-dir", type=Path, help="Optional config override.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-frames", type=int, help="Optional evenly sampled frame limit.")
    parser.add_argument("--background-samples", type=int, default=64)
    parser.add_argument("--embedding-samples", type=int, default=5000,
                        help="Maximum frames for ResNet/PCA analysis; 0 means all.")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--near-duplicate-ssim", type=float, default=0.985)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def resolve_path(value, config_path):
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(args):
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    data = config.get("data", {}) or {}
    if args.training_dir:
        training = args.training_dir.expanduser().resolve()
    elif data.get("training") or data.get("train"):
        training = resolve_path(data.get("training") or data.get("train"), config_path)
    elif data.get("dataset_base"):
        training = resolve_path(data["dataset_base"], config_path) / "train"
    else:
        raise ValueError("Config needs data.dataset_base, data.train, or data.training")
    mask_value = (data.get("mask_types", {}) or {}).get(args.safety_area)
    mask_path = resolve_path(mask_value, config_path) if mask_value else None
    return training, mask_path


def evenly_sample(items, maximum):
    if not maximum or len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, maximum, dtype=int)
    return [items[index] for index in indices]


def scenario_id(path):
    match = SCENARIO_PATTERN.search(path.name)
    return match.group(1) if match else "unknown"


def phash(gray):
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    bits = low > np.median(low[1:])
    return f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"


def global_ssim(first, second):
    first, second = first.astype(np.float64), second.astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    m1, m2 = first.mean(), second.mean()
    v1, v2 = first.var(), second.var()
    covariance = ((first - m1) * (second - m2)).mean()
    return float(((2*m1*m2+c1)*(2*covariance+c2))/((m1*m1+m2*m2+c1)*(v1+v2+c2)))


def build_background(paths, sample_count):
    frames = []
    for path in evenly_sample(paths, min(sample_count, len(paths))):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA))
    if not frames:
        raise ValueError("No readable training images")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def appearance_metrics(paths, background, mask_path, progress):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.is_file() else None
    rows, thumbnails, unreadable = [], [], []
    for index, path in enumerate(paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            unreadable.append(str(path)); continue
        image128 = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image128, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image128, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 80, 160)
        difference = cv2.cvtColor(cv2.absdiff(image128, background), cv2.COLOR_BGR2GRAY)
        valid = np.any(image128 > 5, axis=2)
        if mask is not None and mask.shape == image.shape[:2]:
            roi = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST) > 127
        else:
            roi = np.ones((128, 128), dtype=bool)
        usable = roi & valid
        histogram = np.concatenate([
            cv2.calcHist([image128], [channel], None, [8], [0, 256]).ravel()
            for channel in range(3)
        ]).astype(float)
        histogram /= max(histogram.sum(), 1.0)
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        rows.append({
            "index": len(rows), "path": str(path.resolve()), "uri": path.resolve().as_uri(),
            "filename": path.name, "scenario_id": scenario_id(path),
            "brightness": float(gray.mean()), "contrast": float(gray.std()),
            "mean_saturation": float(hsv[..., 1].mean()),
            "edge_density": float(np.mean(edges > 0)),
            "object_occupancy_proxy": float(np.mean((difference > 25) & roi)),
            "foreground_mask_occupancy": float(np.mean(usable)),
            "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "dark_pixel_fraction": float(np.mean(gray < 30)),
            "saturated_pixel_fraction": float(np.mean(gray > 245)),
            "phash": phash(gray), "color_histogram": histogram.tolist(),
        })
        thumbnails.append(thumb)
        if progress and (index % 1000 == 0 or index == len(paths)):
            print(f"[appearance] {index}/{len(paths)}")
    return rows, np.asarray(thumbnails), unreadable


class EmbeddingDataset(Dataset):
    def __init__(self, paths, transform): self.paths, self.transform = paths, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(image)), index


def semantic_analysis(paths, args, device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    loader = DataLoader(EmbeddingDataset(paths, weights.transforms()),
                        batch_size=args.embedding_batch_size, shuffle=False, num_workers=0)
    embeddings, indices = [], []
    with torch.no_grad():
        for batch_index, (images, batch_indices) in enumerate(loader, start=1):
            vectors = model(images.to(device)).cpu().numpy()
            embeddings.append(vectors); indices.extend(batch_indices.numpy().tolist())
            if args.progress:
                print(f"[embeddings] batch {batch_index}/{len(loader)}")
    matrix = np.concatenate(embeddings)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    components = centered @ eigenvectors[:, -2:]
    cluster_count = max(2, min(args.clusters, len(matrix)))
    cv2.setRNGSeed(42)
    _, clusters, _ = cv2.kmeans(
        matrix.astype(np.float32), cluster_count, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4),
        5, cv2.KMEANS_PP_CENTERS,
    )
    clusters = clusters.ravel()
    nearest_distance = np.empty(len(matrix), dtype=float)
    nearest_index = np.empty(len(matrix), dtype=int)
    for start in range(0, len(matrix), 512):
        similarities = matrix[start:start + 512] @ matrix.T
        row_indices = np.arange(start, min(start + 512, len(matrix)))
        similarities[np.arange(len(row_indices)), row_indices] = -np.inf
        nearest_index[start:start + len(row_indices)] = np.argmax(similarities, axis=1)
        nearest_distance[start:start + len(row_indices)] = 1.0 - np.max(similarities, axis=1)
    return matrix, components, clusters, nearest_distance, nearest_index


def robust_outlier_flags(rows):
    metrics = ["brightness", "contrast", "edge_density", "object_occupancy_proxy",
               "blur_score", "dark_pixel_fraction", "saturated_pixel_fraction"]
    matrix = np.asarray([[row[key] for key in metrics] for row in rows], dtype=float)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    z = np.abs(matrix - median) / np.maximum(1.4826 * mad, 1e-9)
    scores = np.max(z, axis=1)
    for row, score, values in zip(rows, scores, z):
        row["appearance_outlier_score"] = float(score)
        reasons = [metrics[i] for i, value in enumerate(values) if value >= 4.0]
        row["quality_flags"] = ";".join(reasons)


def duplicate_analysis(rows, thumbnails, threshold):
    hashes = {}
    for row in rows: hashes.setdefault(row["phash"], []).append(row["index"])
    features = thumbnails.reshape(len(thumbnails), -1).astype(np.float32)
    features -= features.mean(axis=1, keepdims=True)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
    matches = matcher.knnMatch(features, features, k=min(2, len(features)))
    parent = list(range(len(rows)))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]; value = parent[value]
        return value
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for index, row in enumerate(rows):
        candidates = [match.trainIdx for match in matches[index] if match.trainIdx != index]
        neighbor = int(candidates[0]) if candidates else index
        similarity = global_ssim(thumbnails[index], thumbnails[neighbor])
        row["nearest_visual_index"] = neighbor
        row["nearest_visual_ssim"] = similarity
        row["exact_duplicate_count"] = len(hashes[row["phash"]]) - 1
        if similarity >= threshold: union(index, neighbor)
    roots = [find(index) for index in range(len(rows))]
    unique_roots, root_counts = np.unique(roots, return_counts=True)
    sizes = dict(zip(unique_roots.tolist(), root_counts.tolist()))
    for row, root in zip(rows, roots):
        row["near_duplicate_group"] = root
        row["near_duplicate_group_size"] = sizes[root]
    return len(set(roots)), sum(len(group) - 1 for group in hashes.values())


def write_csv(path, rows):
    excluded = {"uri", "semantic_nearest_uri", "color_histogram"}
    fields = [key for key in rows[0] if key not in excluded]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_html(path, rows, summary, semantic_rows):
    figure = make_subplots(rows=2, cols=3, subplot_titles=(
        "Brightness", "Contrast", "Blur score", "Edge density",
        "Foreground/object occupancy proxy", "PCA semantic embedding"))
    for column, key in enumerate(("brightness", "contrast", "blur_score"), start=1):
        figure.add_trace(go.Histogram(x=[r[key] for r in rows], name=key), row=1, col=column)
    figure.add_trace(go.Histogram(x=[r["edge_density"] for r in rows], name="edge density"), row=2, col=1)
    figure.add_trace(go.Scatter(x=[r["foreground_mask_occupancy"] for r in rows],
                                y=[r["object_occupancy_proxy"] for r in rows], mode="markers",
                                marker={"size": 4, "opacity": .5}, name="occupancy"), row=2, col=2)
    if semantic_rows:
        figure.add_trace(go.Scatter(
            x=[r["pca_x"] for r in semantic_rows], y=[r["pca_y"] for r in semantic_rows],
            mode="markers", marker={"color": [r["semantic_cluster"] for r in semantic_rows],
                                     "colorscale": "Turbo", "size": 6},
            customdata=[[r["filename"], r["scenario_id"], r["uri"], r["semantic_nearest_distance"]] for r in semantic_rows],
            hovertemplate="%{customdata[0]}<br>Scenario %{customdata[1]}<br>Nearest distance %{customdata[3]:.4f}<extra></extra>",
            name="embedding"), row=2, col=3)
    figure.update_layout(height=850, title=f"Normal training-data quality — {escape(summary['safety_area'])}",
                         template="plotly_white", showlegend=False)
    plot_html = figure.to_html(include_plotlyjs=True, full_html=False, div_id="quality-plot")
    mean_histogram = np.mean([row["color_histogram"] for row in rows], axis=0)
    color_figure = go.Figure()
    for channel, color, offset in (("Blue", "blue", 0), ("Green", "green", 8), ("Red", "red", 16)):
        color_figure.add_trace(go.Bar(x=list(range(8)), y=mean_histogram[offset:offset+8],
                                      name=channel, marker_color=color, opacity=.65))
    color_figure.update_layout(title="Mean 8-bin color histogram", barmode="group",
                               xaxis_title="Intensity bin", yaxis_title="Normalized frequency",
                               template="plotly_white")
    color_html = color_figure.to_html(include_plotlyjs=False, full_html=False)
    flagged = sorted(rows, key=lambda row: row["appearance_outlier_score"], reverse=True)[:100]
    flagged_cards = "".join(
        f'<article><img src="{escape(row["uri"])}"><div><b>{escape(row["filename"])}</b><br>'
        f'Scenario {escape(row["scenario_id"])} · score {row["appearance_outlier_score"]:.2f}<br>'
        f'Flags: {escape(row["quality_flags"] or "high combined deviation")}</div></article>'
        for row in flagged
    )
    summary_items = "".join(f"<li><b>{escape(str(key))}:</b> {escape(str(value))}</li>"
                            for key, value in summary.items())
    nearest_cards = "".join(
        f'<article><img src="{escape(row["uri"])}"><div><b>Isolated candidate</b><br>'
        f'{escape(row["filename"])}<br>embedding distance {row["semantic_nearest_distance"]:.4f}</div>'
        f'<img src="{escape(row["semantic_nearest_uri"])}"><div><b>Nearest neighbor</b><br>'
        f'{escape(Path(row["semantic_nearest_path"]).name)}</div></article>'
        for row in sorted(semantic_rows, key=lambda item: item["semantic_nearest_distance"], reverse=True)[:30]
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Normal data quality</title>
<style>body{{font-family:system-ui;background:#f4f6f8;color:#17212b;margin:0}}main{{max-width:1700px;margin:auto;padding:22px}}details{{background:white;padding:0 18px 18px;margin:16px 0;border-radius:10px}}summary{{cursor:pointer;font-size:1.25rem;font-weight:700;padding:18px 0;user-select:none}}details[open]>summary{{border-bottom:1px solid #e2e8f0;margin-bottom:14px}}ul.summary-list{{columns:2}}.guide{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}}.guide div{{border:1px solid #d7dde5;border-radius:8px;padding:12px}}.guide h3{{margin-top:0}}.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px;max-height:900px;overflow:auto}}article{{border:1px solid #d7dde5;border-radius:8px;padding:8px;display:flex;gap:10px}}article img{{width:128px;height:128px;object-fit:contain;background:#111}}</style></head><body><main>
<h1>Normal training-data quality audit — {escape(summary['safety_area'])}</h1>
<p><b>Important:</b> this one-class audit finds visual/statistical outliers for human review. It cannot prove that an image is semantically normal.</p>
<details open><summary>Summary</summary><ul class="summary-list">{summary_items}</ul></details>
<details open><summary>How to interpret this report</summary><div class="guide">
<div><h3>Brightness and contrast</h3><p>A compact distribution is consistent. Separate peaks can represent different lighting, shifts, or operating states. Review extreme tails. A large valid mode should not be removed merely because it differs.</p></div>
<div><h3>Blur and edge density</h3><p>Very low blur score usually means defocus or motion blur. Abruptly low edge density can mean blank/covered frames; unusually high values can indicate noise or a visually different object.</p></div>
<div><h3>Occupancy</h3><p>Foreground/mask occupancy measures usable non-black area. Object occupancy is a background-difference proxy, not object detection. Distinct occupancy groups may correspond to empty, entering, centered, and leaving conveyor states.</p></div>
<div><h3>PCA embedding</h3><p>Nearby points look semantically similar to ResNet. Dense clusters indicate recurring appearances. Small distant clusters and isolated points require review, but are not automatically anomalies. Color denotes cluster ID, not normal/anomalous class.</p></div>
<div><h3>Nearest-neighbor distance</h3><p>A large cosine distance means no similar sampled training image was found. Compare each isolated candidate to its nearest neighbor below. Valid rare normal states should usually be retained; incorrect normal labels should be removed or relabeled.</p></div>
<div><h3>Duplicates and effective uniqueness</h3><p>A low effective-unique fraction means repeated frames dominate training. Downsample large duplicate groups while retaining temporal and appearance transitions. Exact hashes are strict; SSIM groups include near-duplicates.</p></div>
<div><h3>Color histogram</h3><p>Dominant bins summarize the dataset color/intensity balance. Strong unexpected peaks can reveal black padding, overexposure, color shifts, or preprocessing differences.</p></div>
<div><h3>Quality flags</h3><p>An appearance score of 4 or more means at least one metric is roughly four robust deviations from the dataset median. Treat it as a review queue, not an anomaly decision.</p></div>
</div></details>
<details open><summary>Appearance and semantic variability plots</summary>{plot_html}</details>
<details><summary>Mean color histogram</summary>{color_html}</details>
<details><summary>Most isolated semantic samples and nearest neighbors</summary><div class="cards">{nearest_cards or 'Embeddings disabled.'}</div></details>
<details><summary>Top frames requiring review</summary><p>Images are loaded from the dataset and are not embedded.</p><div class="cards">{flagged_cards}</div></details>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    training, mask_path = load_config(args)
    normal_dir = training / args.safety_area / "normal"
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Normal training folder not found: {normal_dir}")
    paths = sorted(path for path in normal_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    paths = evenly_sample(paths, args.max_frames)
    if len(paths) < 2: raise ValueError("At least two images are required")
    output = (args.output_dir or Path("reports") / "normal_training_quality" / args.safety_area).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {len(paths)} images from {normal_dir}")
    background = build_background(paths, args.background_samples)
    rows, thumbnails, unreadable = appearance_metrics(paths, background, mask_path, args.progress)
    robust_outlier_flags(rows)
    effective_unique, exact_duplicates = duplicate_analysis(rows, thumbnails, args.near_duplicate_ssim)
    semantic_rows = []
    if not args.no_embeddings:
        selected = evenly_sample(list(range(len(rows))), args.embedding_samples)
        selected_paths = [Path(rows[index]["path"]) for index in selected]
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        _, pca, clusters, distances, neighbors = semantic_analysis(selected_paths, args, device)
        for local_index, row_index in enumerate(selected):
            row = rows[row_index]
            row.update({"pca_x": float(pca[local_index, 0]), "pca_y": float(pca[local_index, 1]),
                        "semantic_cluster": int(clusters[local_index]),
                        "semantic_nearest_distance": float(distances[local_index]),
                        "semantic_nearest_path": rows[selected[int(neighbors[local_index])]]["path"],
                        "semantic_nearest_uri": rows[selected[int(neighbors[local_index])]]["uri"]})
            semantic_rows.append(row)
    summary = {
        "safety_area": args.safety_area, "images_analyzed": len(rows),
        "unreadable_images": len(unreadable), "scenario_count": len({r["scenario_id"] for r in rows}),
        "appearance_outliers_z>=4": sum(r["appearance_outlier_score"] >= 4 for r in rows),
        "exact_duplicate_excess_frames": exact_duplicates,
        "near_duplicate_effective_unique_frames": effective_unique,
        "effective_unique_fraction": f"{effective_unique/len(rows):.2%}",
        "embedding_images": len(semantic_rows), "mask": str(mask_path) if mask_path else "not configured",
        "semantic_cluster_sizes": (
            {str(cluster): sum(row["semantic_cluster"] == cluster for row in semantic_rows)
             for cluster in sorted({row["semantic_cluster"] for row in semantic_rows})}
            if semantic_rows else "embeddings disabled"
        ),
        "normality_conclusion": "Human review required for flagged frames; one-class statistics cannot prove semantic normality.",
    }
    write_csv(output / "frame_quality_metrics.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(output / "normal_training_quality.html", rows, summary, semantic_rows)
    print(f"HTML report: {output / 'normal_training_quality.html'}")
    print(f"Frame metrics: {output / 'frame_quality_metrics.csv'}")


if __name__ == "__main__": main()
