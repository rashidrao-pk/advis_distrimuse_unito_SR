"""
calibrate_threshold.py
----------------------
Threshold calibration for the VAE-GAN CAD anomaly detection system.
Converted from notebooks: N5_Threshold_train.ipynb + N5_threshold_Calibration.ipynb

Two calibration modes  (--mode)
--------------------------------
  val   Simple, unsupervised.
        Scores the normal-only validation split and derives the threshold
        from the score distribution (max / percentile / mean+n*sigma).
        No ground-truth labels required.
        → Equivalent to compute_threshold.py

  test  Supervised, search-based.
        Scores a labelled test set (normal + anomalous frames), sweeps a
        configurable grid of anomaly-scoring functions, and selects the
        method + threshold that maximises binormal_AUC (or recall).
        Requires a ground-truth CSV (--gt_csv).
        → Equivalent to the N5_threshold_Calibration notebook

Output  (both modes)
------
  results/training/threshold/<safety_area>/
      val_scores_<area>_*.csv            raw per-image val scores  (val mode)
      test_scores_<area>_*.csv           raw per-image test scores (test mode)
      anomaly_metrics_<area>.csv         per-method metrics table  (test mode)
      threshold_<area>.json              final threshold + metadata
  results/training/threshold/
      thresholds_summary.csv             one row per area, all modes

Usage
-----
  # val mode (no labels needed)
  python calibrate_threshold.py --mode val --safety_area RoboArm
  python calibrate_threshold.py --mode val --safety_area ALL

  # test mode (needs labelled test set)
  python calibrate_threshold.py --mode test --safety_area RoboArm \\
      --test_folder /data/test --test_scenarios 13_0 \\
      --gt_csv /data/annotations/scenario_13_0_back_view_annotations.csv

  python calibrate_threshold.py --mode test --safety_area ALL \\
      --test_folder /data/test \\
      --annotations_dir /data/annotations
"""

import os
import csv
import json
import math
import copy
import signal
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, minimum_filter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import utils as ut
import utils_model as utmc
from utils_model import Encoder, Decoder, Discriminator

try:
    from tass_cython_distance import (
        compute_distance_offset as _distance_offset_cython,
        compute_minimization_offset as _minimization_offset_cython,
    )
except ImportError:
    _distance_offset_cython = None
    _minimization_offset_cython = None

# ---------------------------------------------------------------------------
# Stop flag
# ---------------------------------------------------------------------------
_STOP = False

def _handle_sigint(sig, frame):
    global _STOP
    print("\n[INFO] SIGINT — stopping after current area.")
    _STOP = True

signal.signal(signal.SIGINT, _handle_sigint)

# ALL_SAFETY_AREAS = ["RoboArm", "ConvBelt", "PLeft", "PRight"]
ALL_SAFETY_AREAS = ["PRight","PLeft", "RoboArm","ConvBelt"]

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "cf_dataset_epito.yaml"


def taas_variant_tag(variant: str) -> str:
    return "" if variant == "canonical" else f"_taas-{variant}"

# ---------------------------------------------------------------------------
# Shared: anomaly scoring (pure-NumPy, matches Cython version in notebooks)
# ---------------------------------------------------------------------------

def _distance_offset_np(imgA: np.ndarray, imgB: np.ndarray,
                         offset: int) -> np.ndarray:
    """Per-pixel minimum Euclidean distance in a (2*offset+1)^2 neighbourhood."""
    H, W, _ = imgA.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    for di in range(-offset, offset + 1):
        for dj in range(-offset, offset + 1):
            i0a = max(0,  di);  i1a = min(H, H + di)
            i0b = max(0, -di);  i1b = min(H, H - di)
            j0a = max(0,  dj);  j1a = min(W, W + dj)
            j0b = max(0, -dj);  j1b = min(W, W - dj)
            d = np.sqrt(((imgA[i0a:i1a, j0a:j1a] -
                          imgB[i0b:i1b, j0b:j1b]) ** 2).sum(axis=2))
            dist[i0a:i1a, j0a:j1a] = np.minimum(dist[i0a:i1a, j0a:j1a],
                                                   d.astype(np.float32))
    return dist


def _minimized_residual_np(imgA: np.ndarray, imgB: np.ndarray,
                           offset: int) -> np.ndarray:
    """Local minimum filter over the aligned RGB reconstruction residual."""
    delta = imgA - imgB
    distance = np.sqrt(
        np.einsum("ijk,ijk->ij", delta, delta, optimize=False)
    ).astype(np.float32)
    return minimum_filter(
        distance, size=2 * offset + 1, mode="constant", cval=np.inf
    )


def resolve_taas_backend(requested: str, variant: str) -> str:
    available = (
        _distance_offset_cython is not None if variant == "canonical"
        else _minimization_offset_cython is not None
    )
    if requested == "auto":
        return "cython" if available else "numpy"
    if requested == "cython" and not available:
        raise RuntimeError(
            "Cython TAAS backend requested but tass_cython_distance is not built. "
            "Run: python scripts/setup_taas_cython.py build_ext --inplace"
        )
    return requested


def score_pair(imgA: np.ndarray, imgB: np.ndarray,
               offset: int, sigma: float, quantile: float,
               taas_variant: str = "canonical",
               taas_backend: str = "numpy",
               ) -> tuple[float, np.ndarray]:
    """Score a single HWC float32 image pair. Returns (scalar, dist_map)."""
    if taas_variant == "canonical":
        dist = (
            _distance_offset_cython(
                np.ascontiguousarray(imgA, dtype=np.float32),
                np.ascontiguousarray(imgB, dtype=np.float32), offset,
            )
            if taas_backend == "cython"
            else _distance_offset_np(imgA, imgB, offset)
        )
    else:
        delta = imgA - imgB
        aligned = np.sqrt(
            np.einsum("ijk,ijk->ij", delta, delta, optimize=False)
        ).astype(np.float32)
        dist = (
            _minimization_offset_cython(
                np.ascontiguousarray(aligned), offset
            )
            if taas_backend == "cython"
            else minimum_filter(
                aligned, size=2 * offset + 1,
                mode="constant", cval=np.inf,
            )
        )
    if sigma > 0:
        dist = gaussian_filter(dist, sigma=sigma)
    return float(np.quantile(dist, quantile)), dist


def tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor in [-1,1] → (H,W,C) float32 in [0,1]."""
    return (t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            * 0.5 + 0.5)


def score_batch(data_t: torch.Tensor, recon_t: torch.Tensor,
                offset: int, sigma: float, quantile: float,
                taas_variant: str = "canonical",
                taas_backend: str = "numpy",
                ) -> np.ndarray:
    """Return (B,) anomaly scores for a batch."""
    scores = []
    for i in range(data_t.shape[0]):
        a = tensor_to_hwc(data_t[i])
        b = tensor_to_hwc(recon_t[i])
        scores.append(
            score_pair(
                a, b, offset, sigma, quantile, taas_variant, taas_backend
            )[0]
        )
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Shared: dataset helpers
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """Minimal torchvision-free ImageFolder-compatible dataset."""

    EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        self.classes = [path.name for path in class_dirs]
        self.class_to_idx = {
            name: index for index, name in enumerate(self.classes)
        }
        self.samples = []
        for class_dir in class_dirs:
            class_index = self.class_to_idx[class_dir.name]
            self.samples.extend(
                (str(path), class_index)
                for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in self.EXTENSIONS
            )
        self.imgs = self.samples
        if not self.samples:
            raise FileNotFoundError(f"No ImageFolder images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            value = self.transform(image) if self.transform else image.copy()
        return value, label


class _SimpleDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self._ds = ImageFolderDataset(root=root, transform=transform)
        self.imgs = self._ds.imgs
        self.classes = self._ds.classes
        self.class_to_idx = self._ds.class_to_idx
        self.samples = self._ds.samples

    def __len__(self):  return len(self._ds)
    def __getitem__(self, i): return self._ds[i]


class AnnotatedTestDataset(Dataset):
    """Normal/anomalous safety-area crops with explicit binary labels."""

    def __init__(self, samples, transform=None):
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.imgs = self.samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            value = self.transform(image) if self.transform else image.copy()
        return value, label


def _val_transform():
    def transform(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5
    return transform


def load_val_loader(split_json: str, root: str,
                    batch_size: int, num_workers: int) -> tuple:
    base = ImageFolderDataset(root=root)
    with open(split_json, encoding="utf-8") as f:
        info = json.load(f)
    from torch.utils.data import Subset
    ds = _SimpleDataset(root, _val_transform())
    sub_ds = Subset(ds, info["val_indices"])
    loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    return loader, sub_ds, info["val_indices"]


def _normalise_annotation_label(value):
    label = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "normal": "normal",
        "anomalous": "anomalous",
        "anomaly": "anomalous",
        "verify": "verify",
        "intermediate": "verify",
    }
    return aliases.get(label)


def _annotation_scenario_id(csv_path, camera, row_scenario):
    """Resolve ordinary and unified annotation rows to a dataset scenario ID."""
    scenario = str(row_scenario).strip()
    if scenario.lower() not in {"unified", "cumulative", "combined"}:
        return scenario
    csv_path = Path(csv_path)
    prefix = "scenario_"
    suffix = f"_{camera}_annotations.csv"
    if csv_path.name.startswith(prefix) and csv_path.name.endswith(suffix):
        return csv_path.name[len(prefix):-len(suffix)]
    raise ValueError(f"Cannot determine cumulative scenario ID for {csv_path}")


def load_annotation_index(annotation_paths, camera, selected_scenarios=None):
    """Load the current safety-area annotation CSV schema."""
    required = {"scenario_id", "camera", "safety_area", "filename", "label"}
    index = {}
    used_paths = []
    for csv_path in annotation_paths:
        csv_path = Path(csv_path).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Unsupported annotation schema in {csv_path}; missing: "
                + ", ".join(sorted(missing))
            )
        used = False
        for row in frame.to_dict("records"):
            scenario = _annotation_scenario_id(
                csv_path, camera, row["scenario_id"]
            )
            if str(row["camera"]).strip() != camera:
                continue
            if selected_scenarios and scenario not in selected_scenarios:
                continue
            area = str(row["safety_area"]).strip()
            filename = Path(str(row["filename"]).strip()).name
            label = _normalise_annotation_label(row["label"])
            if label is None:
                raise ValueError(
                    f"Unsupported label {row['label']!r} in {csv_path}"
                )
            key = (scenario, area, filename)
            previous = index.get(key)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Conflicting labels for {scenario}/{area}/{filename}: "
                    f"{previous} versus {label}"
                )
            index[key] = label
            used = True
        if used:
            used_paths.append(csv_path)
    return index, used_paths


def load_calibration_scenario_details(annotation_paths, camera, selected_scenarios):
    """Return source scenario descriptions represented by calibration rows."""
    descriptions = {}
    for csv_path in annotation_paths:
        csv_path = Path(csv_path).expanduser().resolve()
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        for row in frame.to_dict("records"):
            if str(row.get("camera", "")).strip() != camera:
                continue
            scenario = _annotation_scenario_id(
                csv_path, camera, row.get("scenario_id", "")
            )
            if selected_scenarios and scenario not in selected_scenarios:
                continue
            source_id = str(row.get("source_scenario_id") or scenario).strip()
            description = str(row.get("scenario_description") or "").strip()
            if source_id and (source_id not in descriptions or description):
                descriptions[source_id] = description
    return [
        {"scenario_id": scenario, "description": descriptions[scenario]}
        for scenario in sorted(
            descriptions,
            key=lambda value: tuple(
                int(part) if part.isdigit() else part
                for part in value.split("_")
            ),
        )
    ]


def _resolve_test_scenario_dirs(test_root, requested_scenarios, areas):
    test_root = Path(test_root).expanduser().resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    # A scenario root was supplied directly: test/13_0.
    if any((test_root / area).is_dir() for area in areas):
        scenario_dirs = {test_root.name: test_root}
    else:
        scenario_dirs = {
            path.name: path
            for path in test_root.iterdir()
            if path.is_dir() and any((path / area).is_dir() for area in areas)
        }
    if requested_scenarios:
        missing = sorted(set(requested_scenarios).difference(scenario_dirs))
        if missing:
            raise ValueError(
                "Requested test scenario(s) not found: " + ", ".join(missing)
            )
        scenario_dirs = {
            scenario: scenario_dirs[scenario] for scenario in requested_scenarios
        }
    if not scenario_dirs:
        raise ValueError(
            f"No test/<scenario>/<safety-area> directories found under {test_root}"
        )
    return scenario_dirs


def discover_annotated_test_samples(
    test_root, area, annotation_paths, camera="back_view",
    requested_scenarios=None,
):
    """Discover labelled crops and verify them against saved annotations."""
    requested_scenarios = list(requested_scenarios or [])
    scenario_dirs = _resolve_test_scenario_dirs(
        test_root, requested_scenarios, ALL_SAFETY_AREAS
    )
    selected = set(scenario_dirs)
    annotation_index, used_annotations = load_annotation_index(
        annotation_paths, camera, selected
    )
    if not used_annotations:
        raise ValueError(
            f"No {camera} annotations matched scenarios: "
            + ", ".join(sorted(selected))
        )

    samples = []
    verify_count = 0
    missing_annotations = []
    mismatched_labels = []
    extensions = ImageFolderDataset.EXTENSIONS
    for scenario, scenario_dir in sorted(scenario_dirs.items()):
        area_dir = scenario_dir / area
        if not area_dir.is_dir():
            continue
        for label, binary_label in (("normal", 0), ("anomalous", 1)):
            class_dir = area_dir / label
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                    continue
                key = (scenario, area, image_path.name)
                annotation_label = annotation_index.get(key)
                if annotation_label is None:
                    missing_annotations.append(image_path)
                elif annotation_label != label:
                    mismatched_labels.append((image_path, annotation_label, label))
                else:
                    samples.append((image_path, binary_label))
        verify_dir = area_dir / "verify"
        if verify_dir.is_dir():
            verify_count += sum(
                path.is_file() and path.suffix.lower() in extensions
                for path in verify_dir.rglob("*")
            )

    if missing_annotations:
        raise ValueError(
            f"{len(missing_annotations)} {area} test image(s) have no matching "
            f"annotation; first: {missing_annotations[0]}"
        )
    if mismatched_labels:
        path, annotation_label, folder_label = mismatched_labels[0]
        raise ValueError(
            f"Annotation/folder label mismatch for {path}: annotation="
            f"{annotation_label}, folder={folder_label}"
        )

    counts = {
        "normal": sum(label == 0 for _, label in samples),
        "anomalous": sum(label == 1 for _, label in samples),
        "verify_excluded": int(verify_count),
    }
    metadata = {
        "test_root": str(Path(test_root).expanduser().resolve()),
        "scenarios": sorted(scenario_dirs),
        "source_scenarios": load_calibration_scenario_details(
            used_annotations, camera, selected
        ),
        "annotation_csvs": [str(path) for path in used_annotations],
        **counts,
    }
    return samples, metadata


def load_test_loader(samples, batch_size: int, num_workers: int) -> tuple:
    ds = AnnotatedTestDataset(samples, _val_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    return loader, ds


def _scenario_artifact_tag(scenarios) -> str:
    values = [str(value).strip() for value in scenarios if str(value).strip()]
    if not values:
        return ""
    safe_values = [
        "".join(char if char.isalnum() or char in "_-" else "-" for char in value)
        for value in values
    ]
    if len(safe_values) <= 3:
        return "_scenario-" + "+".join(safe_values)
    return f"_scenarios-{len(safe_values)}"


# ---------------------------------------------------------------------------
# Shared: model loader
# ---------------------------------------------------------------------------

def load_model_for_area(area: str, params, paths, args, device):
    Enc = Encoder(z_size=params.latent_dims).to(device)
    Dec = Decoder(z_size=params.latent_dims).to(device)
    Dis = Discriminator().to(device)
    optED, optD = utmc.get_optimizers(Enc, Dec, Dis, verbose=False)
    # Match train.py: checkpoints are named model_<safety_area>_<latent_dims>.pt.
    suffix = f"{params.subgroup}_{params.latent_dims}"
    print('-'*100)
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
    print('[model checkpoints] ', paths.path_models)
    print('-'*100)
    history, checkpoint_config = utmc.load_model(
        Enc, Dec, Dis, optED, optD,
        paths.path_models, suffix, device=device,
        verbose=True and args.verbose_level > 0,
    )
    if not history and checkpoint_config is None:
        if args.verbose_level>0:
            print('-'*100, f'\nmodel path -- {os.path.exists(paths.path_models)} - {paths.path_models}\n', '-'*100)
        raise RuntimeError(f"No checkpoint for '{area} - {args.checkpoints}'. Run train.py first.")
    n_epochs = (
        checkpoint_config.get("epochs_trained")
        if isinstance(checkpoint_config, dict)
        else None
    )
    if n_epochs is None:
        n_epochs = len(history)

    Enc.eval(); Dec.eval()
    return Enc, Dec, suffix, int(n_epochs)


def reconstruct(Enc, Dec, data_t: torch.Tensor, device) -> torch.Tensor:
    """Deterministic posterior-mean reconstruction, matching inference."""
    data_t = data_t.to(device)
    with torch.inference_mode():
        mu, _ = Enc(data_t)
        return Dec(mu)


# ---------------------------------------------------------------------------
# VAL MODE helpers
# ---------------------------------------------------------------------------

def _select_threshold_from_scores(scores: np.ndarray, strategy: str,
                                   percentile: float, n_sigma: float) -> float:
    if strategy == "max":
        return float(scores.max())
    elif strategy == "percentile":
        return float(np.percentile(scores, percentile))
    elif strategy == "mean_std":
        return float(scores.mean() + n_sigma * scores.std())
    raise ValueError(f"Unknown strategy: {strategy}")


def run_val_mode(area: str, args, device, out_dir: str) -> dict:
    """Score val split, derive threshold from score distribution."""
    params, paths = _setup_params_paths(area, args)
    
    split_json = os.path.join(
        paths.train_dir_processed_subgroup,
        f"split_4train_1val_{area}.json"
    )
    if not os.path.exists(split_json):
        raise FileNotFoundError(
            f"Split JSON not found: {split_json}\n"
            "Run train.py first to create the split."
        )

    val_loader, val_sub, val_indices = load_val_loader(
        split_json, paths.train_dir_processed_subgroup,
        args.batch_size, args.num_workers
    )
    print(f"[val] {len(val_sub)} val images")

    Enc, Dec, suffix, n_epochs = load_model_for_area(area, params, paths, args, device)

    # Score
    records = []
    global_i = 0
    base_ds = ImageFolderDataset(root=paths.train_dir_processed_subgroup)

    for data_t, _ in tqdm(val_loader, desc=f"Scoring val [{area}]", leave=False):
        recon_t = reconstruct(Enc, Dec, data_t, device)
        scores = score_batch(
            data_t, recon_t, args.offset, args.sigma, args.quantile,
            args.taas_variant, args.taas_backend,
        )
        for b in range(data_t.shape[0]):
            real_idx  = val_indices[global_i]
            img_path  = base_ds.imgs[real_idx][0]
            records.append({
                "file_name":     f"fronttop_{Path(img_path).stem.split('_')[1]}",
                "anomaly_score": float(scores[b]),
                "label":         base_ds.imgs[real_idx][1],
            })
            global_i += 1
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)
    score_csv = os.path.join(
        out_dir, area,
        f"{args.mode}_scores_{area}_{args.threshold_strategy}{args.threshold_percentile}"
        f"_off{args.offset}_sig{args.sigma}_q{args.quantile}"
        f"{taas_variant_tag(args.taas_variant)}.csv"
    )
    os.makedirs(os.path.dirname(score_csv), exist_ok=True)
    df.to_csv(score_csv, index=False)
    print(f"[save] Val scores → {score_csv}")

    tau = _select_threshold_from_scores(
        df.anomaly_score.values, args.threshold_strategy,
        args.threshold_percentile, args.threshold_n_sigma
    )

    summary = _build_summary(area, suffix, n_epochs, args, tau, df, score_csv, "val")
    _save_threshold_json(out_dir, area, summary, args)
    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# TEST MODE helpers
# ---------------------------------------------------------------------------

def _build_scoring_grid(args) -> list:
    """
    Build the list of (name, score_fn, parameters) tuples to sweep.
    Each score_fn takes (data_tensor_BCHW, recon_tensor_BCHW) → np.ndarray (B,).
    """
    fns = []

    # Fixed offset grid
    offset_ls   = [int(x) for x in args.offset_ls.split(",")]
    quantile_ls = [float(x) for x in args.quantile_ls.split(",")]
    sigma_ls    = [float(x) for x in args.sigma_ls.split(",")]

    for offset in offset_ls:
        for quantile in quantile_ls:
            for sigma in sigma_ls:
                prefix = (
                    "TAAS" if args.taas_variant == "canonical"
                    else "TAAS_RESIDUAL_MIN"
                )
                name = f"{prefix}_OFF{offset}-s_{sigma}-q_{quantile}"
                # capture loop vars
                def _fn(d, r, _o=offset, _q=quantile, _s=sigma):
                    return score_batch(
                        d, r, _o, _s, _q, args.taas_variant,
                        args.taas_backend,
                    )
                fns.append((name, _fn, {
                    "offset": offset,
                    "sigma": sigma,
                    "quantile": quantile,
                    "taas_variant": args.taas_variant,
                }))

    return fns


def _compute_scores_for_loader(loader, dataset, Enc, Dec, score_fn,
                                device) -> tuple[list, list, list]:
    """
    Returns (scores, binary_labels, file_names).
    binary_labels: 1 = anomalous, 0 = normal.
    """
    scores, labels, fnames = [], [], []
    global_i = 0

    for data_t, lbl_t in tqdm(loader, desc="Scoring", leave=False, position=2):
        recon_t = reconstruct(Enc, Dec, data_t, device)
        batch_s = score_fn(data_t, recon_t)

        for b in range(data_t.shape[0]):
            img_path = dataset.imgs[global_i][0]
            frame_key = Path(img_path).name
            is_anomalous = int(lbl_t[b].item())
            scores.append(float(batch_s[b]))
            labels.append(is_anomalous)
            fnames.append(frame_key)
            global_i += 1
        torch.cuda.empty_cache()

    return scores, labels, fnames


def _compute_threshold_f1c(labels, scores) -> float:
    """F1-optimal threshold via precision-recall curve."""
    prec, rec, thr = precision_recall_curve(labels, scores)
    if not len(thr):
        return 0.5
    f1 = np.divide(
        2.0 * prec * rec,
        prec + rec,
        out=np.zeros_like(prec, dtype=float),
        where=(prec + rec) > 0,
    )
    # precision_recall_curve returns one extra terminal precision/recall point
    # that has no corresponding threshold.
    idx = min(int(np.nanargmax(f1)), len(thr) - 1)
    return float(thr[idx])


def _binormal_auc(tnv, tpv) -> float:
    if len(tnv) == 0 or len(tpv) == 0:
        return float("nan")
    tn_m, tn_s = np.mean(tnv), np.std(tnv)
    tp_m, tp_s = np.mean(tpv), np.std(tpv)
    denom = math.sqrt(tn_s**2 + tp_s**2)
    return abs(tn_m - tp_m) / denom if denom > 0 else float("nan")


def _evaluate_method(name, scores, labels, threshold,
                     monitor_score, current_best) -> tuple[dict, float, int, float]:
    """
    Compute all metrics for one scoring method.
    Returns (metrics_dict, new_best_score, best_idx_flag, threshold).
    """
    scores  = np.asarray(scores,  dtype=float)
    labels  = np.asarray(labels,  dtype=int)
    preds   = (scores >= threshold).astype(int)

    acc  = accuracy_score (labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score   (labels, preds, zero_division=0)
    f1   = f1_score       (labels, preds, zero_division=0)

    has_both = len(np.unique(labels)) == 2
    auc = roc_auc_score(labels, scores) if has_both else float("nan")

    tnv = scores[(labels == 0) & (scores <  threshold)]
    tpv = scores[(labels == 1) & (scores >= threshold)]
    b_auc = _binormal_auc(tnv, tpv)

    m = dict(Method=name, Accuracy=acc, Precision=prec, Recall=rec,
             F1=f1, AUC=auc, Threshold=threshold, binormal_AUC=b_auc)

    if monitor_score == "binormal_auc":
        score_val = b_auc if not math.isnan(b_auc) else -1
    elif monitor_score == "recall":
        score_val = rec
    else:
        raise ValueError(f"Unknown monitor_score: {monitor_score}")

    is_new_best = score_val > current_best
    return m, score_val if is_new_best else current_best, is_new_best, threshold



def _save_calibration_plots(name, scenario_tag, scores, labels, threshold, params,
                             save_dir: str, destroy: bool = True):
    """Scatter + KDE plot for one scoring method."""
    scores = np.asarray(scores); labels = np.asarray(labels)
    preds  = (scores >= threshold).astype(int)

    tnv = scores[(labels == 0) & (scores <  threshold)]
    tpv = scores[(labels == 1) & (scores >= threshold)]
    fnv = scores[(labels == 1) & (scores <  threshold)]
    fpv = scores[(labels == 0) & (scores >= threshold)]

    acc  = accuracy_score (labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score   (labels, preds, zero_division=0)
    f1   = f1_score       (labels, preds, zero_division=0)
    b_auc = _binormal_auc(tnv, tpv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})

    cat_data = {
        "True Negatives":  (np.where((labels==0)&(preds==0))[0], "blue"),
        "False Negatives": (np.where((labels==1)&(preds==0))[0], "orange"),
        "True Positives":  (np.where((labels==1)&(preds==1))[0], "green"),
        "False Positives": (np.where((labels==0)&(preds==1))[0], "red"),
    }
    for lbl, (idxs, color) in cat_data.items():
        if len(idxs):
            ax1.scatter(idxs, scores[idxs], label=f"{lbl} ({len(idxs)})",
                        alpha=0.6, color=color, s=12)
    ax1.axhline(threshold, color="gray", linestyle="--",
                label=f"tau = {threshold:.4f}")
    ax1.set_title(
        f"{name} | {params.subgroup}\n"
        f"Acc:{acc:.2f} F1:{f1:.2f} Prec:{prec:.2f} Rec:{rec:.2f} bAUC:{b_auc:.2f}"
    )
    ax1.set_xlabel("Index"); ax1.set_ylabel("Anomaly Score")
    ax1.legend(fontsize=7); ax1.grid(True)

    df_kde = pd.DataFrame({
        "value": np.concatenate([tnv, tpv]),
        "group": ["TN"] * len(tnv) + ["TP"] * len(tpv),
    })
    if len(df_kde) and sns is not None:
        sns.kdeplot(data=df_kde, x="value", hue="group", fill=True,
                    common_norm=False, ax=ax2,
                    palette={"TN": "skyblue", "TP": "lightgreen"})
    elif len(df_kde):
        if len(tnv):
            ax2.hist(tnv, bins=30, density=True, alpha=0.45,
                     color="skyblue", label="TN")
        if len(tpv):
            ax2.hist(tpv, bins=30, density=True, alpha=0.45,
                     color="lightgreen", label="TP")
    if len(tnv): ax2.axvline(tnv.mean(), color="blue",  ls="--",
                              label=f"TN mean={tnv.mean():.3f}")
    if len(tpv): ax2.axvline(tpv.mean(), color="green", ls="--",
                              label=f"TP mean={tpv.mean():.3f}")
    handles, lbl_names = ax2.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="none",
                           label=f"bAUC: {b_auc:.4f}"))
    ax2.legend(handles=handles, fontsize=7)
    ax2.set_title("KDE: TN vs TP")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir,
                f"{name.replace(' ', '_')}_{params.subgroup}_plot{scenario_tag}.png"),
                dpi=120, bbox_inches="tight")
    if destroy:
        plt.close(fig)


def _skip_test_calibration_without_normal(
    area, test_metadata, area_out, scenario_tag,
):
    """Record an unsafe anomalous-only area without replacing its threshold."""
    summary = {
        "safety_area": area,
        "mode": "test",
        "status": "skipped",
        "calibration_mode": "anomalous_only_skipped",
        "skip_reason": "no_normal_samples_for_safety_area",
        "threshold": None,
        "threshold_written": False,
        "supervised_metrics_available": False,
        "n_images": int(test_metadata["anomalous"]),
        "n_normal": 0,
        "n_anomalous": int(test_metadata["anomalous"]),
        "n_verify_excluded": int(test_metadata["verify_excluded"]),
        "calibration_scenarios": list(test_metadata["scenarios"]),
        "calibration_source_scenarios": list(
            test_metadata["source_scenarios"]
        ),
        "calibration_data": test_metadata,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(area_out, exist_ok=True)
    report_path = os.path.join(
        area_out,
        f"calibration_skipped_{area}{scenario_tag}_no-normal.json",
    )
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    metrics_path = os.path.join(
        area_out, f"anomaly_metrics_{area}{scenario_tag}.csv"
    )
    pd.DataFrame([{
        "Method": None,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "AUC": None,
        "Threshold": None,
        "binormal_AUC": None,
        "Status": "skipped_no_normal_samples",
    }]).to_csv(metrics_path, index=False)

    print(
        f"[skip] {area}: normal=0, anomalous={test_metadata['anomalous']}, "
        f"verify={test_metadata['verify_excluded']}. A safe threshold cannot "
        "be calibrated from anomalous samples alone."
    )
    print(f"[skip] Existing threshold files were not modified.")
    print(f"[skip] Report → {report_path}")
    return summary


def _run_normal_only_test_calibration(
    area, args, device, test_loader, test_ds, test_metadata, Enc, Dec,
    suffix, n_epochs, area_out, out_dir, scenario_tag,
):
    """Calibrate from normal test frames when an area has no anomalies."""
    parameters = {
        "offset": args.offset,
        "sigma": args.sigma,
        "quantile": args.quantile,
        "taas_variant": args.taas_variant,
    }
    prefix = "TAAS" if args.taas_variant == "canonical" else "TAAS_RESIDUAL_MIN"
    score_name = f"{prefix}_OFF{args.offset}-s_{args.sigma}-q_{args.quantile}"

    def score_fn(data, reconstruction):
        return score_batch(
            data, reconstruction, args.offset, args.sigma, args.quantile,
            args.taas_variant, args.taas_backend,
        )

    scores, labels, filenames = _compute_scores_for_loader(
        test_loader, test_ds, Enc, Dec, score_fn, device
    )
    # The scoring helper returns Python lists during real DataLoader runs.
    # Convert before vector comparisons and before using distribution methods
    # such as ndarray.max() in the normal-only threshold strategies.
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if np.any(labels != 0):
        raise RuntimeError(
            f"Normal-only fallback for {area} received non-normal labels"
        )
    threshold = _select_threshold_from_scores(
        scores, args.threshold_strategy,
        args.threshold_percentile, args.threshold_n_sigma,
    )
    df_scores = pd.DataFrame({
        "file_name": filenames,
        "anomaly_score": scores,
        "label": labels,
    })
    score_csv = os.path.join(
        area_out,
        f"test_scores_{area}{scenario_tag}_normal-only_{score_name}.csv",
    )
    df_scores.to_csv(score_csv, index=False)

    metrics_csv = os.path.join(
        area_out, f"anomaly_metrics_{area}{scenario_tag}.csv"
    )
    pd.DataFrame([{
        "Method": score_name,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "AUC": None,
        "Threshold": threshold,
        "binormal_AUC": None,
        "Status": "not_applicable_no_anomalous_samples",
    }]).to_csv(metrics_csv, index=False)

    summary = _build_summary(
        area, suffix, n_epochs, args, threshold,
        df_scores, score_csv, "test",
        score_parameters=parameters,
        threshold_strategy_override=args.threshold_strategy,
        calibration_mode="normal_only_fallback",
        fallback_reason="no_anomalous_samples_for_safety_area",
        supervised_metrics_available=False,
        best_method=None,
        best_binormal_auc=None,
        best_recall=None,
        best_f1=None,
        threshold_selection=args.threshold_strategy,
        selection_metric=None,
        threshold_n_sigma=(
            args.threshold_n_sigma
            if args.threshold_strategy == "mean_std" else None
        ),
        calibration_data=test_metadata,
        n_normal=test_metadata["normal"],
        n_anomalous=0,
        n_verify_excluded=test_metadata["verify_excluded"],
    )
    _save_threshold_json(out_dir, area, summary, args)
    print(f"\n{'='*70}")
    print(f"[result] Safety area   : {area}")
    print(f"[result] Test scenario : {', '.join(test_metadata['scenarios'])}")
    print("[result] Calibration   : normal-only fallback (no anomalies)")
    print(f"[result] Strategy      : {args.threshold_strategy}")
    print(f"[result] Score         : {score_name}")
    print(f"[result] Threshold     : {threshold:.6f}")
    print("[result] F1/AUC        : not applicable")
    print(f"{'='*70}")
    return summary


def run_test_mode(area: str, args, device, out_dir: str) -> dict:
    """
    Score labelled test set, sweep scoring functions, pick best threshold.
    """
    # ── Dataset and annotation preflight ──────────────────────────────────
    samples, test_metadata = discover_annotated_test_samples(
        args.test_dir,
        area,
        args.annotation_paths,
        camera=args.camera,
        requested_scenarios=args.test_scenarios,
    )
    test_loader, test_ds = load_test_loader(
        samples, args.batch_size, args.num_workers
    )
    print(
        f"[test] {len(test_ds)} usable images for {area} | "
        f"normal={test_metadata['normal']} | "
        f"anomalous={test_metadata['anomalous']} | "
        f"verify excluded={test_metadata['verify_excluded']}"
    )
    print(f"[test] scenarios: {', '.join(test_metadata['scenarios'])}")
    for source in test_metadata["source_scenarios"]:
        description = source["description"] or "description unavailable"
        print(f"[test] source scenario {source['scenario_id']}: {description}")

    area_out = os.path.join(out_dir, area)
    scenario_tag = _scenario_artifact_tag(test_metadata["scenarios"])
    if test_metadata["normal"] == 0:
        return _skip_test_calibration_without_normal(
            area, test_metadata, area_out, scenario_tag
        )

    # ── Setup model ───────────────────────────────────────────────────────
    params, paths = _setup_params_paths(area, args)
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)

    Enc, Dec, suffix, n_epochs = load_model_for_area(area, params, paths, args, device)

    # ── Output dirs ───────────────────────────────────────────────────────
    plot_dir = os.path.join(area_out, "calibration_plots")
    os.makedirs(area_out, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if test_metadata["anomalous"] == 0:
        print(
            f"[fallback] {area} has no anomalous samples; calibrating from "
            f"{test_metadata['normal']} normal samples with "
            f"strategy={args.threshold_strategy}. Supervised F1/AUC metrics "
            "are not applicable."
        )
        return _run_normal_only_test_calibration(
            area, args, device, test_loader, test_ds, test_metadata, Enc, Dec,
            suffix, n_epochs, area_out, out_dir, scenario_tag,
        )

    csv_metrics = os.path.join(
        area_out, f"anomaly_metrics_{area}{scenario_tag}.csv"
    )
    csv_header  = ["Method", "Accuracy", "Precision", "Recall",
                   "F1", "AUC", "Threshold", "binormal_AUC"]

    # ── Score sweep ───────────────────────────────────────────────────────
    scoring_grid = _build_scoring_grid(args)
    print(f"[sweep] {len(scoring_grid)} scoring configurations")

    best_score      = -1.0
    best_idx        = 0
    best_threshold  = None
    best_name       = None
    best_parameters = None
    all_metrics     = []

    with open(csv_metrics, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_header)

    params.epochs_loaded = n_epochs

    for fn_idx, (name, score_fn, score_parameters) in enumerate(
            tqdm(scoring_grid, desc=f"Calibrating [{area}]", position=1)):
        if _STOP:
            break

        scores, labels, fnames = _compute_scores_for_loader(
            test_loader, test_ds, Enc, Dec, score_fn, device
        )

        if len(np.unique(labels)) < 2:
            print(f"[warn] {name}: only one label class — skipping.")
            continue

        # Threshold selection (f1c = F1-optimising via PRC)
        if args.threshold_method == "f1c":
            threshold = _compute_threshold_f1c(labels, scores)
        else:
            raise ValueError(f"Unknown --threshold_method: {args.threshold_method}")

        metrics, best_score, is_best, threshold = _evaluate_method(
            name, scores, labels, threshold,
            args.monitor_score, best_score
        )
        if is_best:
            best_idx       = fn_idx
            best_threshold = threshold
            best_name      = name
            best_parameters = score_parameters

        all_metrics.append(metrics)

        with open(csv_metrics, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([metrics[k] for k in csv_header])

        _save_calibration_plots(name,scenario_tag,scores, labels, threshold,
                                 params, plot_dir, destroy=True)

    if best_name is None:
        raise RuntimeError("No valid scoring method found (check GT labels).")

    # ── Rename best-method plot ───────────────────────────────────────────
    old_f = os.path.join(plot_dir,
                f"{best_name.replace(' ','_')}_{area}_plot.png")
    new_f = os.path.join(plot_dir,
                f"BEST-{best_name.replace(' ','_')}_{area}_plot.png")
    if os.path.exists(old_f):
        os.replace(old_f, new_f)

    # ── Save raw test scores for the best method ──────────────────────────
    best_fn   = scoring_grid[best_idx][1]
    scores_b, labels_b, fnames_b = _compute_scores_for_loader(
        test_loader, test_ds, Enc, Dec, best_fn, device
    )
    df_scores = pd.DataFrame({"file_name": fnames_b,
                               "anomaly_score": scores_b,
                               "label": labels_b})
    score_csv = os.path.join(area_out,
        f"test_scores_{area}{scenario_tag}_{best_name.replace(' ','_')}.csv")
    df_scores.to_csv(score_csv, index=False)

    # ── Metrics table ─────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics).sort_values(
        "binormal_AUC", ascending=False)
    df_metrics.to_csv(csv_metrics, index=False)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"[result] Best method   : {best_name}")
    print(f"[result] Threshold     : {best_threshold:.6f}")
    print(f"[result] Safety area   : {area}")
    print(f"[result] Test scenario : {', '.join(test_metadata['scenarios'])}")
    print(f"[result] Epochs trained: {n_epochs}")
    print(f"[result] Monitor score : {args.monitor_score}")
    print(f"{'='*70}")

    df_best = df_metrics[df_metrics.Method == best_name].iloc[0]
    print(f"  Accuracy  : {df_best.Accuracy:.3f}")
    print(f"  Precision : {df_best.Precision:.3f}")
    print(f"  Recall    : {df_best.Recall:.3f}")
    print(f"  F1        : {df_best.F1:.3f}")
    print(f"  binAUC    : {df_best.binormal_AUC:.3f}")

    summary = _build_summary(
        area, suffix, n_epochs, args, best_threshold,
        df_scores, score_csv, "test",
        score_parameters=best_parameters,
        best_method=best_name,
        best_binormal_auc=float(df_best.binormal_AUC),
        best_recall=float(df_best.Recall),
        best_f1=float(df_best.F1),
        threshold_selection=args.threshold_method,
        selection_metric=args.monitor_score,
        calibration_data=test_metadata,
        n_normal=test_metadata["normal"],
        n_anomalous=test_metadata["anomalous"],
        n_verify_excluded=test_metadata["verify_excluded"],
    )
    _save_threshold_json(out_dir, area, summary, args)
    return summary


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _setup_params_paths(area: str, args):
    params, paths = ut.get_params_paths()
    paths         = ut.get_paths(paths, verbose=False)

    # ``utils.get_paths`` only initialises path_datasets_main for a small set
    # of legacy host names.  Config-driven runs (notably macOS hosts) must not
    # depend on that machine-name lookup.  Seed the value required by
    # get_dataset_version, then replace its derived paths below with the
    # explicit YAML paths.
    dataset_base = Path(args.dataset_base).expanduser().resolve()
    if not hasattr(paths, "path_datasets_main"):
        paths.path_datasets_main = str(dataset_base.parent)

    params.subgroup      = area
    params.latent_dims   = args.latent_dims
    params.exp_type      = args.exp_type
    params.subgroup_mask = "mask"
    params.batch_size    = args.batch_size
    paths, params = ut.get_dataset_version(
        paths, params,
        dataset_version = args.dataset_version,
        dataset_type    = args.dataset_type,
        mask_image_name = args.mask_image_name,
        subgroup        = area,
        verbose         = False,
    )

    paths.path_datasets = str(dataset_base)
    paths.path_dataset_selected = str(dataset_base)
    if args.training_dir:
        training_dir = Path(args.training_dir).expanduser().resolve()
        paths.train_dir = str(training_dir)
        paths.train_dir_processed = str(training_dir)
        paths.train_dir_subgroup = str(training_dir / area)
        paths.train_dir_processed_subgroup = str(training_dir / area)
    if args.testing_dir:
        testing_dir = Path(args.testing_dir).expanduser().resolve()
        paths.test_dir = str(testing_dir)
        paths.test_dir_processed = str(testing_dir)
        paths.test_dir_subgroup = str(testing_dir / area)
        paths.test_dir_processed_subgroup = str(testing_dir / area)
    if args.masks_dir:
        paths.mask_dir = str(Path(args.masks_dir).expanduser().resolve())

    params = ut.get_parameters_by_experiment(params, verbose=False)
    paths.path_codes_cloud = paths.path_codes
    paths.path_codes_main  = os.path.join(paths.path_codes, "scripts")
    # paths.path_models      = os.path.join(paths.path_codes_main, "results", "models")
    paths.path_models      = os.path.join(paths.path_codes_main, args.checkpoints)
    paths.path_results_cloud = os.path.join(paths.path_codes_cloud, 'scripts/results')
    os.makedirs(paths.path_codes_main, exist_ok=True)
    os.makedirs(paths.path_models, exist_ok=True)
    return params, paths


def _build_summary(area, suffix, n_epochs, args, tau, df_scores,
                   score_csv, mode, score_parameters=None,
                   threshold_strategy_override=None, **extra) -> dict:
    score_parameters = score_parameters or {
        "offset": args.offset,
        "sigma": args.sigma,
        "quantile": args.quantile,
        "taas_variant": args.taas_variant,
    }
    offset = int(score_parameters["offset"])
    sigma = float(score_parameters["sigma"])
    quantile = float(score_parameters["quantile"])
    taas_variant = score_parameters.get("taas_variant", args.taas_variant)
    score_prefix = "TAAS" if taas_variant == "canonical" else "TAAS_RESIDUAL_MIN"
    threshold_strategy = threshold_strategy_override or (
        args.threshold_strategy if mode == "val" else args.threshold_method
    )
    s = {
        "safety_area":        area,
        "mode":               mode,
        "suffix":             suffix,
        "epochs_trained":     n_epochs,
        "threshold":          float(tau),
        "threshold_strategy": threshold_strategy,
        "threshold_percentile": (
            args.threshold_percentile
            if threshold_strategy not in {"max", "f1c"} else None
        ),
        "offset":             offset,
        "sigma":              sigma,
        "quantile":           quantile,
        "taas_variant":       taas_variant,
        "taas_backend":       args.taas_backend,
        "score_func":         f"{score_prefix}_OFF{offset}-s_{sigma}-q_{quantile}",
        "reconstruction_mode": "posterior_mean",
        "score_max":          float(df_scores.anomaly_score.max()),
        "score_mean":         float(df_scores.anomaly_score.mean()),
        "score_std":          float(df_scores.anomaly_score.std()),
        "score_p99":          float(np.percentile(df_scores.anomaly_score, 99)),
        "n_images":           len(df_scores),
        "score_csv":          score_csv,
        "computed_at":        datetime.now().isoformat(timespec="seconds"),
    }
    s.update(extra)
    calibration_data = s.get("calibration_data") or {}
    if mode == "test":
        s["calibration_scenarios"] = list(
            calibration_data.get("scenarios") or []
        )
        s["calibration_source_scenarios"] = list(
            calibration_data.get("source_scenarios") or []
        )
    return s


def _save_threshold_json(out_dir: str, area: str, summary: dict, args):
    area_dir = os.path.join(out_dir, area)
    os.makedirs(area_dir, exist_ok=True)
    strategy = summary["threshold_strategy"]
    strategy_tag = (
        strategy
        if summary.get("threshold_percentile") is None
        else f"{strategy}{summary['threshold_percentile']}"
    )
    filename_tail = (
        f"_off{summary['offset']}_sig{summary['sigma']}_q{summary['quantile']}"
        f"{taas_variant_tag(summary['taas_variant'])}.json"
    )
    json_path = os.path.join(
        area_dir,
        f"threshold_{args.mode}_{area}_{strategy_tag}"
        f"{filename_tail}",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Threshold JSON → {json_path}")

    # Keep the conventional name above as the active inference-compatible
    # threshold.  Test calibration also gets an immutable, traceable copy.
    scenario_tag = _scenario_artifact_tag(
        summary.get("calibration_scenarios") or []
    )
    if summary.get("mode") == "test" and scenario_tag:
        archived_path = os.path.join(
            area_dir,
            f"threshold_{area}_{strategy_tag}{scenario_tag}{filename_tail}",
        )
        with open(archived_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[save] Scenario threshold archive → {archived_path}")
        return archived_path
    return json_path


def _print_summary(s: dict):
    print('-'*50)
    print(f"\n[result] {'safety_area':<25} {s['safety_area']}")
    print(f"[result] {'mode':<25} {s['mode']}")
    print(f"[result] {'threshold':<25} {s['threshold']:.6f}")
    print(f"[result] {'score_max':<25} {s['score_max']:.6f}")
    print(f"[result] {'score_mean':<25} {s['score_mean']:.6f}")
    print(f"[result] {'n_images':<25} {s['n_images']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model_config(config_file: Path) -> dict:
    """Load model settings used by training from a YAML configuration."""
    config_path = config_file.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Config must contain a 'models' mapping: {config_path}")
    if not models.get("checkpoints"):
        raise ValueError(f"Config must define 'models.checkpoints': {config_path}")
    return models

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Threshold calibration — val mode or supervised test mode.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Mode & areas ──────────────────────────────────────────────────────
    p.add_argument("--mode", default="val", choices=["val", "test"],
                   help=(
                       "val  : Unsupervised — derive threshold from normal val scores.\n"
                       "test : Supervised — sweep scoring methods on labelled test set\n"
                       "       and pick best threshold by binormal_AUC. If an area\n"
                       "       has no anomalies, use a normal-only fallback."
                   ))
    p.add_argument("--safety_area", default="RoboArm",
                   help="Area to calibrate. 'ALL' processes all areas.")

    # ── Model / dataset ───────────────────────────────────────────────────
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training YAML config (default: {DEFAULT_CONFIG}).",
    )
    p.add_argument("--dataset_version",  default="v6")
    p.add_argument(
        "--dataset_type",
        default=None,
        help=(
            "Optional dataset subdirectory below --dataset_version. "
            "Omit it (or pass 'None') to use the version directory directly."
        ),
    )
    p.add_argument("--mask_image_name",  default=3015, type=int)
    p.add_argument(
        "--latent_dims", default=None, type=int,
        help="Latent size; defaults to models.latent_dims from --config.",
    )
    p.add_argument("--exp_type",         default="E3")
    p.add_argument("--save_path_type",   default="cloud",
                   choices=["cloud", "local"])
    p.add_argument("--verbose_level",       default=0,      type=int, choices=[0, 1, 2])
    p.add_argument("--batch_size",       default=32,   type=int)
    p.add_argument("--num_workers",      default=0,    type=int)
    p.add_argument("--save_figures", action="store_true", default=False,
                   help="Save per-frame detection PNG figures.")
    # ── Test-mode inputs ──────────────────────────────────────────────────
    p.add_argument(
        "--test_folder", "--test-folder", default=None,
        help=(
            "[test mode] Test root using test/<scenario>/<area>/<class>. "
            "Defaults to data.testing from --config. It may also point "
            "directly to one scenario directory."
        ),
    )
    p.add_argument(
        "--test_scenarios", "--test-scenarios",
        "--test_scenario", "--test-scenario",
        dest="test_scenarios", nargs="+",
        help="[test mode] Scenario IDs to combine. Default: all under the test root.",
    )
    p.add_argument("--camera", default="back_view")
    p.add_argument(
        "--annotations_dir", "--annotations-dir", type=Path,
        default=Path("reports/safety_area_annotations/saved_annotation"),
        help="Directory containing scenario annotation CSV files.",
    )
    p.add_argument(
        "--checkpoints",
        default=None,
        help="Checkpoint directory; overrides models.checkpoints from --config.",
    )
    p.add_argument(
        "--gt_csv", "--annotation-csv", default=None,
        help=(
            "[test mode] Optional single annotation CSV using the current "
            "scenario_id/camera/safety_area/filename/label schema. This overrides "
            "automatic scenario CSV selection from --annotations_dir."
        ),
    )

    # ── Anomaly score params (both modes) ─────────────────────────────────
    p.add_argument("--offset",   default=1,   type=int) # 1
    p.add_argument("--sigma",    default=1.0, type=float)
    p.add_argument("--quantile", default=0.99, type=float) #0.99
    p.add_argument(
        "--taas_variant", "--taas-variant",
        choices=("canonical", "minimization"), default="canonical",
        help="TAAS score definition used for calibration (default: canonical).",
    )
    p.add_argument(
        "--taas_backend", "--taas-backend",
        choices=("auto", "cython", "numpy"), default="auto",
        help="TAAS implementation used while scoring calibration images.",
    )

    # ── Val-mode threshold strategies ─────────────────────────────────────
    p.add_argument("--threshold_strategy",   default="max",
                   choices=["max", "percentile", "mean_std"],
                   help=(
                       "How to derive a normal-only threshold in val mode or "
                       "when a test safety area has no anomalous samples."
                   ))
    p.add_argument("--threshold_percentile", default=99.0, type=float)
    p.add_argument("--threshold_n_sigma",    default=3.0,  type=float)

    # ── Test-mode scoring grid & selection ────────────────────────────────
    p.add_argument("--offset_ls",   default="1,2,3",
                   help="[test mode] Comma-separated offsets to sweep.")
    p.add_argument("--quantile_ls", default="1.0,0.999,0.99,0.98",
                   help="[test mode] Comma-separated quantiles to sweep.")
    p.add_argument("--sigma_ls",    default="0.5,1.0,1.5",
                   help="[test mode] Comma-separated sigmas to sweep.")
    p.add_argument("--threshold_method", default="f1c",
                   choices=["f1c"],
                   help="[test mode] How to find threshold from PRC.")
    p.add_argument("--monitor_score",    default="binormal_auc",
                   choices=["binormal_auc", "recall"],
                   help="[test mode] Metric to maximise when picking best method.")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default=None,
                   help="Override output dir (default: results/training/threshold).")


    args = p.parse_args(argv)
    args.taas_backend = resolve_taas_backend(
        args.taas_backend, args.taas_variant
    )
    try:
        model_config = load_model_config(args.config)
        with args.config.expanduser().resolve().open("r", encoding="utf-8") as stream:
            full_config = yaml.safe_load(stream) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        p.error(str(exc))

    args.latent_dims = args.latent_dims or model_config.get("latent_dims", 64)
    checkpoint_value = args.checkpoints or model_config["checkpoints"]
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent.parent / checkpoint_path
    args.checkpoints = str(checkpoint_path.resolve())
    data_config = full_config.get("data") or {}
    args.dataset_base = Path(data_config.get("dataset_base", ".")).expanduser()
    args.training_dir = data_config.get("training")
    args.testing_dir = data_config.get("testing")
    args.masks_dir = data_config.get("masks")
    if args.test_folder is None:
        args.test_folder = args.testing_dir or str(args.dataset_base / "test")
    return args


def resolve_test_annotation_paths(
    *, gt_csv, annotations_dir, test_scenarios, camera, project_root,
):
    """Resolve explicit or scenario-derived annotation CSV paths."""
    project_root = Path(project_root).expanduser().resolve()
    if gt_csv:
        path = Path(gt_csv).expanduser()
        path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Annotation CSV not found: {path}")
        return [path]

    directory = Path(annotations_dir).expanduser()
    if not directory.is_absolute():
        directory = project_root / directory
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {directory}")

    if test_scenarios:
        paths = []
        for value in test_scenarios:
            scenario = str(value).strip()
            if not scenario or Path(scenario).name != scenario:
                raise ValueError(f"Invalid test scenario ID: {value!r}")
            path = directory / f"scenario_{scenario}_{camera}_annotations.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Annotation CSV for scenario {scenario!r} and camera "
                    f"{camera!r} not found: {path}"
                )
            paths.append(path)
        return paths

    paths = sorted(directory.glob(f"scenario_*_{camera}_annotations.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No {camera} annotation CSVs found in {directory}"
        )
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  |  mode={args.mode}")
    print(
        f"[TAAS] variant={args.taas_variant} | backend={args.taas_backend}"
    )

    areas = ALL_SAFETY_AREAS if args.safety_area.upper() == "ALL" \
            else [args.safety_area]

    # Resolve output root once (needs at least one area for the path)
    params, paths = ut.get_params_paths()
    paths = ut.get_paths(paths, verbose=False)
    paths.path_codes_main = os.path.join(paths.path_codes, "scripts")
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
    print(args.dataset_version)
    out_dir = args.output_dir or os.path.join(
        paths.path_codes, "results",args.dataset_version, "thresholds"
    )
    test_path = Path(args.test_folder).expanduser()
    if not test_path.is_absolute():
        test_path = args.dataset_base / test_path
    args.test_dir = str(test_path.resolve())

    args.annotation_paths = []
    if args.mode == "test":
        args.annotation_paths = resolve_test_annotation_paths(
            gt_csv=args.gt_csv,
            annotations_dir=args.annotations_dir,
            test_scenarios=args.test_scenarios,
            camera=args.camera,
            project_root=Path(__file__).resolve().parent.parent,
        )
        print(
            "[annotations] "
            + ", ".join(path.name for path in args.annotation_paths)
        )
    
    if args.verbose_level>1:
        print('-'*100)
        print(f'PATHS - \npath_datasets_main:{paths.path_datasets_main} \ntest_folder:{args.test_folder} \ntest_dir: {paths.test_dir}')
        print(f'TEst Folder FILE WILL BE LOADED FROM \t {os.path.exists(args.test_dir)} - {args.test_dir}')
        print(f'ANNOTATION CSV FILES \t{len(args.annotation_paths)}')
        print('-'*100)

    os.makedirs(out_dir, exist_ok=True)
    print(f"[output] {out_dir}")

    all_summaries = []
    for i, area in enumerate(areas):
        if _STOP:
            print(f"[INFO] Stopping — skipping: {areas[i:]}")
            break
        print(f"\n[progress] {i+1}/{len(areas)}: {area}")
        print('='*100)
        if args.mode == "val":
            summary = run_val_mode(area, args, device, out_dir)
        else:
            summary = run_test_mode(area, args, device, out_dir)
        all_summaries.append(summary)

    # ── Combined summary CSV ──────────────────────────────────────────────
    if all_summaries:
        summary_csv = os.path.join(
            out_dir,
            (
                f"thresholds_summary_{args.mode}_{args.threshold_strategy}"
                f"{args.threshold_percentile}_off{args.offset}_sig{args.sigma}"
                f"_q{args.quantile}{taas_variant_tag(args.taas_variant)}.csv"
                # if args.mode == "val"
                # else f"thresholds_summary_test_{args.threshold_method}.csv"
            ),
        )
        pd.DataFrame(all_summaries).to_csv(summary_csv, index=False)
        print(f"\n[save] Summary CSV → {summary_csv}")

        df_s = pd.DataFrame(all_summaries)
        cols = ["safety_area", "mode", "epochs_trained",
                "threshold", "score_max", "score_mean", "n_images"]
        if "best_method" in df_s.columns:
            cols += ["best_method", "best_binormal_auc", "best_recall"]
        print("\n" + "="*80)
        print("CALIBRATION SUMMARY")
        print("="*80)
        print(df_s[[c for c in cols if c in df_s.columns]].to_string(index=False))
        print("="*80)

    print("\nDone.")


if __name__ == "__main__":
    main()
