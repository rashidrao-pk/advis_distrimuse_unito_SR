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
      --test_dir /data/test/RoboArm \\
      --gt_csv   /data/annotations/anom_metadata.csv

  python calibrate_threshold.py --mode test --safety_area ALL \\
      --test_dir /data/test \\
      --gt_csv   /data/annotations/anom_metadata.csv
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

import utils_CAD as utc
import utils_model_CAD as utmc
from utils_model_CAD import Encoder, Decoder, Discriminator

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
ALL_SAFETY_AREAS = ["PRight","PLeft", "RoboArm"]

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


def score_pair(imgA: np.ndarray, imgB: np.ndarray,
               offset: int, sigma: float, quantile: float
               ) -> tuple[float, np.ndarray]:
    """Score a single HWC float32 image pair. Returns (scalar, dist_map)."""
    dist = _distance_offset_np(imgA, imgB, offset)
    if sigma > 0:
        dist = gaussian_filter(dist, sigma=sigma)
    return float(np.quantile(dist, quantile)), dist


def tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor in [-1,1] → (H,W,C) float32 in [0,1]."""
    return (t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            * 0.5 + 0.5)


def score_batch(data_t: torch.Tensor, recon_t: torch.Tensor,
                offset: int, sigma: float, quantile: float
                ) -> np.ndarray:
    """Return (B,) anomaly scores for a batch."""
    scores = []
    for i in range(data_t.shape[0]):
        a = tensor_to_hwc(data_t[i])
        b = tensor_to_hwc(recon_t[i])
        scores.append(score_pair(a, b, offset, sigma, quantile)[0])
    return np.array(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Shared: dataset helpers
# ---------------------------------------------------------------------------

class _SimpleDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self._ds = datasets.ImageFolder(root=root, transform=transform)
        self.imgs = self._ds.imgs
        self.classes = self._ds.classes
        self.class_to_idx = self._ds.class_to_idx
        self.samples = self._ds.samples

    def __len__(self):  return len(self._ds)
    def __getitem__(self, i): return self._ds[i]


def _val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_val_loader(split_json: str, root: str,
                    batch_size: int, num_workers: int) -> tuple:
    base = datasets.ImageFolder(root=root)
    with open(split_json, encoding="utf-8") as f:
        info = json.load(f)
    from torch.utils.data import Subset
    ds = _SimpleDataset(root, _val_transform())
    sub_ds = Subset(ds, info["val_indices"])
    loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    return loader, sub_ds, info["val_indices"]


def load_test_loader(test_dir: str, batch_size: int,
                     num_workers: int) -> tuple:
    ds = _SimpleDataset(test_dir, _val_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    return loader, ds


# ---------------------------------------------------------------------------
# Shared: model loader
# ---------------------------------------------------------------------------

def load_model_for_area(area: str, params, paths, args, device):
    Enc = Encoder(z_size=params.latent_dims).to(device)
    Dec = Decoder(z_size=params.latent_dims).to(device)
    Dis = Discriminator().to(device)
    optED, optD = utmc.get_optimizers(Enc, Dec, Dis, verbose=False)
    suffix, paths = utc.get_create_results_path(
        area, params,args, paths,
        save_path_type=args.save_path_type,
        dir="scripts/results", verbose=False,
    )
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
    history = utmc.load_model(Enc, Dec, Dis, optED, optD,
                               paths, suffix, device=device, verbose=True and args.verbose_level>0)
    if not history:
        if args.verbose_level>0:
            print('-'*100, f'\nmodel path -- {os.path.exists(paths.path_models)} - {paths.path_models}\n', '-'*100)
        raise RuntimeError(f"No checkpoint for '{area} - {args.checkpoints}'. Run train.py first.")
    Enc.eval(); Dec.eval()
    return Enc, Dec, suffix, len(history)


def reconstruct(Enc, Dec, data_t: torch.Tensor, device) -> torch.Tensor:
    data_t = data_t.to(device)
    with torch.no_grad():
        mu, logvar = Enc(data_t)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return Dec(z)


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
    base_ds = datasets.ImageFolder(root=paths.train_dir_processed_subgroup)

    for data_t, _ in tqdm(val_loader, desc=f"Scoring val [{area}]", leave=False):
        recon_t = reconstruct(Enc, Dec, data_t, device)
        scores  = score_batch(data_t, recon_t, args.offset, args.sigma, args.quantile)
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
        f"val_scores_{area}_off{args.offset}_sig{args.sigma}_q{args.quantile}.csv"
    )
    os.makedirs(os.path.dirname(score_csv), exist_ok=True)
    df.to_csv(score_csv, index=False)
    print(f"[save] Val scores → {score_csv}")

    tau = _select_threshold_from_scores(
        df.anomaly_score.values, args.threshold_strategy,
        args.threshold_percentile, args.threshold_n_sigma
    )

    summary = _build_summary(area, suffix, n_epochs, args, tau, df, score_csv, "val")
    _save_threshold_json(out_dir, area, summary)
    _print_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# TEST MODE helpers
# ---------------------------------------------------------------------------

def _build_scoring_grid(args) -> list:
    """
    Build the list of (name, score_fn) tuples to sweep.
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
                name = f"OFF-o{offset}-q{quantile}-s{sigma}"
                # capture loop vars
                def _fn(d, r, _o=offset, _q=quantile, _s=sigma):
                    return score_batch(d, r, _o, _s, _q)
                fns.append((name, _fn))

    return fns


def _compute_scores_for_loader(loader, dataset, Enc, Dec, score_fn,
                                good_classname: str, gt_frame_set: set,
                                device) -> tuple[list, list, list]:
    """
    Returns (scores, binary_labels, file_names).
    binary_labels: 1 = anomalous, 0 = normal.
    gt_frame_set: set of frame keys that are anomalous (from GT CSV).
    """
    scores, labels, fnames = [], [], []
    global_i = 0

    for data_t, lbl_t in tqdm(loader, desc="Scoring", leave=False, position=2):
        recon_t = reconstruct(Enc, Dec, data_t, device)
        batch_s = score_fn(data_t, recon_t)

        for b in range(data_t.shape[0]):
            img_path   = dataset.imgs[global_i][0]
            frame_key  = f"fronttop_{Path(img_path).stem.split('_')[1]}"
            is_anomalous = (frame_key in gt_frame_set)
            scores.append(float(batch_s[b]))
            labels.append(int(is_anomalous))
            fnames.append(frame_key)
            global_i += 1
        torch.cuda.empty_cache()

    return scores, labels, fnames


def _compute_threshold_f1c(labels, scores) -> float:
    """F1-optimal threshold via precision-recall curve."""
    prec, rec, thr = precision_recall_curve(labels, scores)
    pr_product = np.multiply(prec, rec)
    idx = pr_product.argmax()
    if idx == 0:
        return float(thr[0]) if len(thr) else 0.5
    if idx >= len(thr):
        return float(thr[-1]) if len(thr) else 0.5
    return float(0.5 * (thr[idx] + thr[idx - 1]))


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


def _save_calibration_plots(name, scores, labels, threshold, params,
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
    if len(df_kde):
        sns.kdeplot(data=df_kde, x="value", hue="group", fill=True,
                    common_norm=False, ax=ax2,
                    palette={"TN": "skyblue", "TP": "lightgreen"})
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
                f"{name.replace(' ', '_')}_{params.subgroup}_plot.png"),
                dpi=120, bbox_inches="tight")
    if destroy:
        plt.close(fig)


def run_test_mode(area: str, args, device, out_dir: str) -> dict:
    """
    Score labelled test set, sweep scoring functions, pick best threshold.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    params, paths = _setup_params_paths(area, args)
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)

    Enc, Dec, suffix, n_epochs = load_model_for_area(area, params, paths, args, device)

    # ── Ground-truth CSV ──────────────────────────────────────────────────
    if not args.gt_csv_path or not os.path.exists(args.gt_csv_path):
        raise FileNotFoundError(
            f"--gt_csv is required for test mode and must exist.\n"
            f"Given: {args.gt_csv_path}"
        )
    df_gt    = pd.read_csv(args.gt_csv_path)
    # Build set of anomalous frame keys for this area
    gt_anom  = df_gt[
        (df_gt["component"] == area) &
        (df_gt["component_anomaly"] == "ANOMALOUS")
    ]
    gt_frame_set = set(f"fronttop_{fn}" for fn in gt_anom["frame_no"].astype(str))
    print(f"[gt]  {len(gt_frame_set)} anomalous frames for {area}")

    # ── Test loader ───────────────────────────────────────────────────────
    test_dir = args.test_dir
    if not test_dir:
        raise ValueError("--test_dir must be provided for test mode.")
    # Accept either a per-area subdirectory or a shared root
    area_test_dir = os.path.join(test_dir, area)
    if not os.path.isdir(area_test_dir):
        area_test_dir = test_dir
    test_loader, test_ds = load_test_loader(area_test_dir, args.batch_size,
                                             args.num_workers)
    good_classname = test_ds.classes[0]
    print(f"[test] {len(test_ds)} images | good class: '{good_classname}'")

    # ── Output dirs ───────────────────────────────────────────────────────
    area_out = os.path.join(out_dir, area)
    plot_dir = os.path.join(area_out, "calibration_plots")
    os.makedirs(area_out, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    csv_metrics = os.path.join(area_out, f"anomaly_metrics_{area}.csv")
    csv_header  = ["Method", "Accuracy", "Precision", "Recall",
                   "F1", "AUC", "Threshold", "binormal_AUC"]

    # ── Score sweep ───────────────────────────────────────────────────────
    scoring_grid = _build_scoring_grid(args)
    print(f"[sweep] {len(scoring_grid)} scoring configurations")

    best_score      = -1.0
    best_idx        = 0
    best_threshold  = None
    best_name       = None
    all_metrics     = []

    with open(csv_metrics, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_header)

    params.epochs_loaded = n_epochs

    for fn_idx, (name, score_fn) in enumerate(
            tqdm(scoring_grid, desc=f"Calibrating [{area}]", position=1)):
        if _STOP:
            break

        scores, labels, fnames = _compute_scores_for_loader(
            test_loader, test_ds, Enc, Dec, score_fn,
            good_classname, gt_frame_set, device
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

        all_metrics.append(metrics)

        with open(csv_metrics, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([metrics[k] for k in csv_header])

        _save_calibration_plots(name, scores, labels, threshold,
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
        test_loader, test_ds, Enc, Dec, best_fn,
        good_classname, gt_frame_set, device
    )
    df_scores = pd.DataFrame({"file_name": fnames_b,
                               "anomaly_score": scores_b,
                               "label": labels_b})
    score_csv = os.path.join(area_out,
        f"test_scores_{area}_{best_name.replace(' ','_')}.csv")
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
    print(f"[result] Epochs trained: {n_epochs}")
    print(f"[result] Monitor score : {args.monitor_score}")
    print(f"{'='*70}")

    df_best = df_metrics[df_metrics.Method == best_name].iloc[0]
    print(f"  Accuracy  : {df_best.Accuracy:.3f}")
    print(f"  Precision : {df_best.Precision:.3f}")
    print(f"  Recall    : {df_best.Recall:.3f}")
    print(f"  F1        : {df_best.F1:.3f}")
    print(f"  binAUC    : {df_best.binormal_AUC:.3f}")

    summary = _build_summary(area, suffix, n_epochs, args, best_threshold,
                              df_scores, score_csv, "test",
                              best_method=best_name,
                              best_binormal_auc=float(df_best.binormal_AUC),
                              best_recall=float(df_best.Recall),
                              best_f1=float(df_best.F1))
    _save_threshold_json(out_dir, area, summary)
    return summary


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _setup_params_paths(area: str, args):
    params, paths = utc.get_params_paths()
    paths         = utc.get_paths(paths, verbose=False)
    params.subgroup      = area
    params.latent_dims   = args.latent_dims
    params.exp_type      = args.exp_type
    params.subgroup_mask = "mask"
    params.batch_size    = args.batch_size
    paths, params = utc.get_dataset_version(
        paths, params,
        dataset_version = args.dataset_version,
        dataset_type    = args.dataset_type,
        mask_image_name = args.mask_image_name,
        subgroup        = area,
        verbose         = False,
    )
    params = utc.get_parameters_by_experiment(params, verbose=False)
    paths.path_codes_cloud = paths.path_codes
    paths.path_codes_main  = os.path.join(paths.path_codes, "scripts")
    # paths.path_models      = os.path.join(paths.path_codes_main, "results", "models")
    paths.path_models      = os.path.join(paths.path_codes_main, args.checkpoints)
    paths.path_results_cloud = os.path.join(paths.path_codes_cloud, 'scripts/results')
    os.makedirs(paths.path_codes_main, exist_ok=True)
    os.makedirs(paths.path_models, exist_ok=True)
    return params, paths


def _build_summary(area, suffix, n_epochs, args, tau, df_scores,
                   score_csv, mode, **extra) -> dict:
    s = {
        "safety_area":        area,
        "mode":               mode,
        "suffix":             suffix,
        "epochs_trained":     n_epochs,
        "threshold":          float(tau),
        "threshold_strategy": args.threshold_strategy if mode == "val"
                              else args.threshold_method,
        "offset":             args.offset,
        "sigma":              args.sigma,
        "quantile":           args.quantile,
        "score_max":          float(df_scores.anomaly_score.max()),
        "score_mean":         float(df_scores.anomaly_score.mean()),
        "score_std":          float(df_scores.anomaly_score.std()),
        "score_p99":          float(np.percentile(df_scores.anomaly_score, 99)),
        "n_images":           len(df_scores),
        "score_csv":          score_csv,
        "computed_at":        datetime.now().isoformat(timespec="seconds"),
    }
    s.update(extra)
    return s


def _save_threshold_json(out_dir: str, area: str, summary: dict):
    area_dir = os.path.join(out_dir, area)
    os.makedirs(area_dir, exist_ok=True)
    json_path = os.path.join(area_dir, f"threshold_{area}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Threshold JSON → {json_path}")


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

def parse_args():
    p = argparse.ArgumentParser(
        description="Threshold calibration — val mode or supervised test mode.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Mode & areas ──────────────────────────────────────────────────────
    p.add_argument("--mode", default="val", choices=["val", "test"],
                   help=(
                       "val  : Unsupervised — derive threshold from normal val scores.\n"
                       "test : Supervised — sweep scoring methods on labelled test set\n"
                       "       and pick best threshold by binormal_AUC."
                   ))
    p.add_argument("--safety_area", default="RoboArm",
                   help="Area to calibrate. 'ALL' processes all areas.")

    # ── Model / dataset ───────────────────────────────────────────────────
    p.add_argument("--dataset_version",  default="v2")
    p.add_argument("--dataset_type",     default="refined")
    p.add_argument("--mask_image_name",  default=3015, type=int)
    p.add_argument("--latent_dims",      default=64,   type=int)
    p.add_argument("--exp_type",         default="E3")
    p.add_argument("--save_path_type",   default="cloud",
                   choices=["cloud", "local"])
    p.add_argument("--verbose_level",       default=0,      type=int, choices=[0, 1, 2])
    p.add_argument("--batch_size",       default=32,   type=int)
    p.add_argument("--num_workers",      default=0,    type=int)
    p.add_argument("--save_figures", action="store_true", default=False,
                   help="Save per-frame detection PNG figures.")
    # ── Test-mode inputs ──────────────────────────────────────────────────
    p.add_argument("--test_folder", default=r'v2/ camera1_20251210_151444_fallen_operator/test/operator_fall',
                   help="[test mode] Root of labelled test ImageFolder.\n"
                        "May contain per-area sub-dirs or be a flat folder.")
    p.add_argument("--checkpoints",
                   default="scripts/checkpoints_33", 
                   choices=["scripts/checkpoints_32", 
                            "scripts/checkpoints_33"])
    p.add_argument("--gt_csv",   default="scripts/data/annotations/anom_metadata_operator_fall.csv",
                   help="[test mode] Ground-truth CSV with columns:\n"
                        "  frame_no, component, component_anomaly\n"
                        "  (component_anomaly ∈ {ANOMALOUS, NORMAL})")

    # ── Anomaly score params (both modes) ─────────────────────────────────
    p.add_argument("--offset",   default=1,   type=int)
    p.add_argument("--sigma",    default=1.0, type=float)
    p.add_argument("--quantile", default=1.0, type=float)

    # ── Val-mode threshold strategies ─────────────────────────────────────
    p.add_argument("--threshold_strategy",   default="max",
                   choices=["max", "percentile", "mean_std"],
                   help="[val mode] How to derive threshold from val scores.")
    p.add_argument("--threshold_percentile", default=99.0, type=float)
    p.add_argument("--threshold_n_sigma",    default=3.0,  type=float)

    # ── Test-mode scoring grid & selection ────────────────────────────────
    p.add_argument("--offset_ls",   default="1,2,3",
                   help="[test mode] Comma-separated offsets to sweep.")
    p.add_argument("--quantile_ls", default="1.0,0.999,0.99",
                   help="[test mode] Comma-separated quantiles to sweep.")
    p.add_argument("--sigma_ls",    default="0,0.5,1.0,1.5",
                   help="[test mode] Comma-separated sigmas to sweep.")
    p.add_argument("--threshold_method", default="f1c",
                   choices=["f1c"],
                   help="[test mode] How to find threshold from PRC.")
    p.add_argument("--monitor_score",    default="binormal_auc",
                   choices=["binormal_auc", "recall"],
                   help="[test mode] Metric to maximise when picking best method.")

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", default=None,
                   help="Override output dir (default: scripts/results/training/threshold).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  |  mode={args.mode}")

    areas = ALL_SAFETY_AREAS if args.safety_area.upper() == "ALL" \
            else [args.safety_area]

    # Resolve output root once (needs at least one area for the path)
    params, paths = utc.get_params_paths()
    paths = utc.get_paths(paths, verbose=False)
    paths.path_codes_main = os.path.join(paths.path_codes, "scripts")
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
    out_dir = args.output_dir or os.path.join(
        paths.path_codes_main, "results", "threshold"
    )
    args.test_dir = os.path.join(paths.path_datasets_main, args.test_folder)
    args.gt_csv_path = os.path.join(os.getcwd(), args.gt_csv)
    
    if args.verbose_level>1:
        print('-'*100)
        print(f'PATHS - \npath_datasets_main:{paths.path_datasets_main} \ntest_folder:{args.test_folder} \ntest_dir: {paths.test_dir}')
        print(f'TEst Folder FILE WILL BE LOADED FROM \t {os.path.exists(args.test_dir)} - {args.test_dir}')
        print(f'CSV FILE WILL BE LOADED FROM \t{os.path.exists(args.gt_csv_path)} - {args.gt_csv_path}')
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
        summary_csv = os.path.join(out_dir, f"thresholds_summary_{args.mode}_{args.threshold_strategy}.csv")
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
