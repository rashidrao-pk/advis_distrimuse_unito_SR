"""
infer.py
--------
Anomaly-detection inference script for the VAE-GAN CAD system.
Converted from notebook: N6_test_using_threshold_train.ipynb

Data sources  (--data_source)
------------------------------
  preprocessed   Pre-cropped/resized images in an ImageFolder directory.
  raw            Full-resolution images that are masked + cropped on-the-fly.
  video          One or more pre-recorded video files (.avi / .mp4).
  ipcam          Live RTSP / HTTP IP-camera stream (frame-by-frame).

Output
------
  results/inference/<run_name>/
      detections_<idx>.png           per-frame visualisation (if --save_figures)
      csv/component_level.csv        per-component, per-frame scores & status
      csv/frame_level.csv            per-frame aggregate status
      evaluation/confmat_<area>.png  confusion matrices (if --gt_csv is given)
      evaluation/metrics_by_area.csv

Usage examples
--------------
  # preprocessed images
  python infer.py --data_source preprocessed --input_dir /data/test/RoboArm

  # raw images, multi-area
  python infer.py --data_source raw --input_dir /data/test/frames --safety_area ALL

  # video file
  python infer.py --data_source video --input_video /data/test_video/fronttop.avi

  # live IP camera
  python infer.py --data_source ipcam --camera_url rtsp://192.168.1.10/stream

  # with ground-truth evaluation
  python infer.py --data_source preprocessed --input_dir /data/test \\
                  --gt_csv /data/annotations/anom_metadata.csv
"""

import os
import cv2
import csv
import json
import math
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")           # headless-safe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, square
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score, classification_report
)

import utils_CAD as utc
import utils_model_CAD as utmc
from utils_model_CAD import Encoder, Decoder, Discriminator

# ---------------------------------------------------------------------------
# Global stop flag (Ctrl+C stops cleanly after the current frame)
# ---------------------------------------------------------------------------
_STOP = False

def _handle_sigint(sig, frame):
    global _STOP
    print("\n[INFO] SIGINT — stopping after current frame.")
    _STOP = True

signal.signal(signal.SIGINT, _handle_sigint)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_SAFETY_AREAS    = ["RoboArm", "ConvBelt", "PLeft", "PRight"]
COMPONENT_FULL_NAME = {
    "PLeft":    "Pallet Left",
    "PRight":   "Pallet Right",
    "RoboArm":  "Robotic Arm",
    "ConvBelt": "Conveyor Belt",
}

# ---------------------------------------------------------------------------
# Anomaly scoring  (pure-NumPy, matches Cython version in the notebook)
# ---------------------------------------------------------------------------

def _compute_distance_offset_np(imgA: np.ndarray, imgB: np.ndarray,
                                  offset: int) -> np.ndarray:
    H, W, _ = imgA.shape
    dist     = np.full((H, W), np.inf, dtype=np.float32)
    for di in range(-offset, offset + 1):
        for dj in range(-offset, offset + 1):
            i0a = max(0,  di);  i1a = min(H, H + di)
            i0b = max(0, -di);  i1b = min(H, H - di)
            j0a = max(0,  dj);  j1a = min(W, W + dj)
            j0b = max(0, -dj);  j1b = min(W, W - dj)
            diff = imgA[i0a:i1a, j0a:j1a] - imgB[i0b:i1b, j0b:j1b]
            d    = np.sqrt((diff ** 2).sum(axis=2)).astype(np.float32)
            dist[i0a:i1a, j0a:j1a] = np.minimum(dist[i0a:i1a, j0a:j1a], d)
    return dist


def compute_anomaly_score_pair(imgA: np.ndarray, imgB: np.ndarray,
                                offset: int, sigma: float,
                                quantile: float) -> tuple[float, np.ndarray]:
    """
    Score a single HWC float32 image pair.
    Returns (scalar_score, distance_map).
    """
    dist = _compute_distance_offset_np(imgA, imgB, offset)
    if sigma > 0:
        dist = gaussian_filter(dist, sigma=sigma)
    return float(np.quantile(dist, quantile)), dist


def tensor_to_hwc_float32(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor in [-1,1]  →  (H,W,C) float32 in [0,1]."""
    return (t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            * 0.5 + 0.5)


# ---------------------------------------------------------------------------
# Threshold loader
# ---------------------------------------------------------------------------

def load_threshold(threshold_dir: str, safety_area: str) -> float:
    """
    Load the threshold from the JSON written by compute_threshold.py.
    Falls back to the max val score from the CSV if the JSON is absent.
    """
    json_path = os.path.join(threshold_dir, safety_area,
                             f"threshold_{safety_area}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        tau = float(d["threshold"])
        print(f"[threshold] {safety_area}: {tau:.6f}  (from {json_path})")
        return tau

    # fallback: scan for a val scores CSV
    area_dir = os.path.join(threshold_dir, safety_area)
    csvs = list(Path(area_dir).glob("val_scores_*.csv")) if os.path.isdir(area_dir) else []
    if csvs:
        df = pd.read_csv(csvs[0])
        tau = float(df["anomaly_score"].max())
        print(f"[threshold] {safety_area}: {tau:.6f}  (max of {csvs[0].name})")
        return tau

    raise FileNotFoundError(
        f"No threshold file found for '{safety_area}' in {threshold_dir}.\n"
        "Run compute_threshold.py first."
    )


# ---------------------------------------------------------------------------
# Model loader — loads all requested safety-area models into a dict
# ---------------------------------------------------------------------------

def load_models(safety_areas, params, paths, args, device) -> dict:
    """
    Returns:
        models_dict[area] = {'encoder': Enc, 'decoder': Dec}
        thresholds[area]  = float
    """
    import copy
    models_dict = {}
    thresholds  = {}

    Enc_base = Encoder(z_size=params.latent_dims).to(device)
    Dec_base = Decoder(z_size=params.latent_dims).to(device)
    Dis_base = Discriminator().to(device)

    for area in safety_areas:
        print(f"[model] Loading: {area}")
        params.subgroup = area
        suffix, paths = utc.get_create_results_path(
            area, params,args, paths,
            save_path_type = args.save_path_type,
            dir            = "scripts/results",
            verbose        = False,
        )

        enc = copy.deepcopy(Enc_base)
        dec = copy.deepcopy(Dec_base)
        dis = copy.deepcopy(Dis_base)
        optED, optD = utmc.get_optimizers(enc, dec, dis, verbose=False)

        paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
        history = utmc.load_model(enc, dec, dis, optED, optD,
                                   paths, suffix, device=device, verbose=False)
        if len(history) == 0:
            if args.verbose_level>0:
                print('-'*100, f'\nmodel path -- {os.path.exists(paths.path_models)} - {paths.path_models}\n', '-'*100)
            raise RuntimeError(
                f"No checkpoint found for '{area}' (suffix={suffix}).\n"
                "Run train.py first."
            )
        enc.eval(); dec.eval()
        models_dict[area] = {"encoder": enc, "decoder": dec,
                              "suffix": suffix, "epochs": len(history)}

        thresholds[area] = load_threshold(args.threshold_dir, area)
        print(f"  epochs={len(history)}  tau={thresholds[area]:.6f}")

    return models_dict, thresholds


# ---------------------------------------------------------------------------
# Component mask + crop transform builder
# ---------------------------------------------------------------------------

def build_component_transform(component: str, paths, params, mask_image_name: int):
    """Load the binary mask and build a crop+resize+normalise transform."""
    mask_name = (f"{paths.dataset_type}_{mask_image_name}_{component}_"
                 f"{params.subgroup_mask}_ext.png")
    mask_path = os.path.join(paths.mask_dir, mask_name)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Component mask not found: {mask_path}")

    mask_arr = np.array(Image.open(mask_path).convert("L"))
    mask_bool = (mask_arr > 128)

    masked_cropper = utc.MaskedCrop(subgroup=component, mask=mask_bool)

    transform = transforms.Compose([
        masked_cropper,
        transforms.Resize(params.target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform, masked_cropper, mask_bool


# ---------------------------------------------------------------------------
# Data source adapters
# All yield (frame_tensor [3,H,W] in [-1,1],  frame_idx,  frame_path_or_none)
# ---------------------------------------------------------------------------

class _PreprocessedSource:
    """
    Images already cropped/resized — directly normalised, one per file.
    Uses torchvision ImageFolder.
    """
    def __init__(self, input_dir: str):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = datasets.ImageFolder(root=input_dir, transform=transform)
        self.loader  = DataLoader(ds, batch_size=1, shuffle=False,
                                  num_workers=0, drop_last=False)
        self.dataset = ds
        print(f"[data] Preprocessed: {len(ds)} images from {input_dir}")

    def __iter__(self):
        for idx, (img_t, _) in enumerate(self.loader):
            path = self.dataset.imgs[idx][0]
            yield img_t.squeeze(0), idx, path

    def __len__(self):
        return len(self.dataset)


class _RawImageSource:
    """
    Full-resolution images.  Masking/cropping is done per-component
    *inside* the inference loop, so here we just load the raw RGB frames.
    """
    def __init__(self, input_dir: str):
        transform = transforms.Compose([transforms.ToTensor()])
        ds = datasets.ImageFolder(root=input_dir, transform=transform)
        self.loader  = DataLoader(ds, batch_size=1, shuffle=False,
                                  num_workers=0, drop_last=False)
        self.dataset = ds
        print(f"[data] Raw images: {len(ds)} frames from {input_dir}")

    def __iter__(self):
        for idx, (img_t, _) in enumerate(self.loader):
            path = self.dataset.imgs[idx][0]
            yield img_t.squeeze(0), idx, path

    def __len__(self):
        return len(self.dataset)


class _VideoSource:
    """Stream frames from a pre-recorded video file."""
    def __init__(self, video_path: str, max_frames: int = None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.cap        = cv2.VideoCapture(video_path)
        self.total      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_frames = max_frames
        self._to_tensor = transforms.ToTensor()
        print(f"[data] Video: {self.total} frames  →  {video_path}")

    def __iter__(self):
        idx = 0
        while True:
            if _STOP:
                break
            if self.max_frames and idx >= self.max_frames:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            # BGR → RGB → float tensor [0,1], then normalise to [-1,1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = self._to_tensor(frame_rgb)                  # [3,H,W] [0,1]
            t = t * 2.0 - 1.0                               # [-1,1]
            yield t, idx, None
            idx += 1
        self.cap.release()

    def __len__(self):
        return self.total if not self.max_frames else min(self.total, self.max_frames)


class _IPCamSource:
    """
    Live IP-camera stream via RTSP/HTTP URL.
    Runs until Ctrl+C or --max_frames is reached.
    """
    def __init__(self, camera_url: str, max_frames: int = None):
        self.cap        = cv2.VideoCapture(camera_url)
        self.max_frames = max_frames
        self._to_tensor = transforms.ToTensor()
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera stream: {camera_url}")
        print(f"[data] IP camera: {camera_url}")

    def __iter__(self):
        idx = 0
        while True:
            if _STOP:
                break
            if self.max_frames and idx >= self.max_frames:
                break
            ok, frame = self.cap.read()
            if not ok:
                print("[warn] Frame grab failed — retrying...")
                time.sleep(0.05)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = self._to_tensor(frame_rgb) * 2.0 - 1.0
            yield t, idx, None
            idx += 1
        self.cap.release()

    def __len__(self):
        return self.max_frames or float("inf")


def build_source(args):
    src = args.data_source.lower()
    if src == "preprocessed":
        return _PreprocessedSource(os.path.join(args.input_dir, args.safety_area))
    elif src == "raw":
        return _RawImageSource(args.input_dir)
    elif src == "video":
        return _VideoSource(args.input_video, args.max_frames)
    elif src == "ipcam":
        return _IPCamSource(args.camera_url, args.max_frames)
    else:
        raise ValueError(f"Unknown --data_source: '{src}'. "
                         "Choose from: preprocessed | raw | video | ipcam")


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def infer_frame(frame_t: torch.Tensor,
                safety_areas: list,
                models_dict: dict,
                thresholds: dict,
                component_transforms: dict,      # area → (transform, cropper, mask_bool)
                args, device, data_source: str) -> dict:
    """
    Run all safety-area models on a single frame tensor (3,H,W) in [-1,1].
    Returns per-component results dict.
    """
    results = {}

    # frame in [0,1] numpy for visualisation / masking
    frame_np01 = ((frame_t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()

    for area in safety_areas:
        enc        = models_dict[area]["encoder"]
        dec        = models_dict[area]["decoder"]
        tau        = thresholds[area]
        transform, cropper, mask_bool = component_transforms[area]

        # ── Build masked input ────────────────────────────────────────────
        if data_source == "preprocessed":
            # Already cropped — use frame directly (assume single-area mode)
            input_tensor = frame_t.unsqueeze(0).to(device)
        else:
            # Mask out everything except this component, then crop + resize
            mask_3ch   = np.stack([mask_bool.astype(np.float32)] * 3, axis=-1)
            masked_np  = frame_np01 * mask_3ch
            input_pil  = TF.to_pil_image(
                torch.from_numpy(masked_np).permute(2, 0, 1).float()
            )
            input_tensor = transform(input_pil).unsqueeze(0).to(device)

        # ── Reconstruct ───────────────────────────────────────────────────
        with torch.no_grad():
            mu, logvar   = enc(input_tensor)
            z            = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            recon_tensor = dec(z)

        # ── Score ─────────────────────────────────────────────────────────
        orig_hwc  = tensor_to_hwc_float32(input_tensor.squeeze(0))
        recon_hwc = tensor_to_hwc_float32(recon_tensor.squeeze(0))
        score, diff_map = compute_anomaly_score_pair(
            orig_hwc, recon_hwc,
            args.offset, args.sigma, args.quantile
        )
        norm_score   = score / tau
        is_anomalous = norm_score > 1.0

        results[area] = {
            "score":       score,
            "norm_score":  norm_score,
            "threshold":   tau,
            "is_anomalous":is_anomalous,
            "status":      "UNEXPECTED" if is_anomalous else "normal",
            "orig_hwc":    orig_hwc,
            "recon_hwc":   recon_hwc,
            "diff_map":    diff_map,
            "cropper":     cropper,
        }

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

CMAP_ANOMALY = LinearSegmentedColormap.from_list("anom", [
    (0.00, "white"),
    (0.25, "lightblue"),
    (0.35, "coral"),
    (0.50, "red"),
    (1.00, "purple"),
])


def _build_full_maps(frame_t, area_results, safety_areas):
    """Paste per-component diff maps and recon images back onto full canvas."""
    H = frame_t.shape[1]; W = frame_t.shape[2]
    full_diff  = np.zeros((H, W),      dtype=np.float32)
    full_recon = np.zeros((H, W, 3),   dtype=np.float32)
    for area in safety_areas:
        r       = area_results[area]
        cropper = r["cropper"]
        y1, y2  = cropper.y1, cropper.y2
        x1, x2  = cropper.x1, cropper.x2
        ch, cw  = y2 - y1, x2 - x1
        diff_r  = cv2.resize(r["diff_map"],  (cw, ch), interpolation=cv2.INTER_LINEAR)
        recon_r = cv2.resize(r["recon_hwc"], (cw, ch), interpolation=cv2.INTER_LINEAR)
        full_diff [y1:y2, x1:x2]    = np.maximum(full_diff[y1:y2, x1:x2], diff_r)
        full_recon[y1:y2, x1:x2, :] = np.maximum(full_recon[y1:y2, x1:x2, :], recon_r)
    return full_diff, full_recon


def save_detection_figure(frame_t, area_results, safety_areas, thresholds,
                           mask_bools, save_path, idx, dpi=150):
    """
    Save a 3-panel detection figure:
      (A) Input + component boundary overlays + status labels
      (B) Anomaly heat-map
      (C) Reconstructed image
    """
    frame_np01 = ((frame_t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    tau_max    = max(thresholds.values())

    full_diff, full_recon = _build_full_maps(frame_t, area_results, safety_areas)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_in, ax_am, ax_re = axes

    ax_in.imshow(frame_np01)
    ax_am.imshow(full_diff, cmap=CMAP_ANOMALY, vmin=0, vmax=2 * tau_max)
    ax_re.imshow(full_recon)

    contour_overlay = np.zeros((*full_diff.shape[:2], 4), dtype=np.float32)

    for area in safety_areas:
        r           = area_results[area]
        mask_bool   = mask_bools[area]
        is_anom     = r["is_anomalous"]
        norm_score  = r["norm_score"]

        bounds    = find_boundaries(mask_bool, mode="thick")
        thick     = dilation(bounds, square(4 if is_anom else 1))
        color     = (1, 0, 0, 1) if is_anom else (0, 0, 0, 1)
        contour_overlay[thick] = color

        ys, xs = np.where(mask_bool)
        if len(ys):
            cx    = float(np.mean(xs))
            cy    = float(np.min(ys))
            label = f"{'unexpected' if is_anom else 'normal'}:{norm_score:.2f}"
            props = dict(facecolor="white", alpha=0.7,
                         edgecolor="red" if is_anom else "green",
                         boxstyle="round,pad=0.2")
            for ax in (ax_in, ax_am):
                ax.text(cx, cy, label,
                        color="red" if is_anom else "black",
                        fontsize=12, ha="center", va="top", bbox=props)

    for ax in (ax_in, ax_am, ax_re):
        ax.imshow(contour_overlay, alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])

    ax_in.set_title("(A) Input + detections", fontsize=13)
    ax_am.set_title("(B) Anomaly Map",         fontsize=13)
    ax_re.set_title("(C) Reconstructed",       fontsize=13)

    plt.subplots_adjust(wspace=0.05)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"detections_{idx:06d}.png"),
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation (vs ground-truth CSV)
# ---------------------------------------------------------------------------

def _binary_cm_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    n  = tn + fp + fn + tp
    acc  = (tp + tn) / n       if n          else 0.0
    prec = tp / (tp + fp)      if (tp + fp)  else 0.0
    rec  = tp / (tp + fn)      if (tp + fn)  else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    spec = tn / (tn + fp)      if (tn + fp)  else 0.0
    return dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                accuracy=acc, precision=prec, recall=rec,
                f1=f1, specificity=spec,
                balanced_accuracy=(rec + spec) / 2)


def evaluate(df_results: pd.DataFrame, df_gt: pd.DataFrame,
             eval_dir: str, safety_areas: list):
    """
    Merge inference results with ground-truth annotations and compute metrics.
    Saves per-area confusion matrix PNGs and a metrics CSV.
    """
    os.makedirs(eval_dir, exist_ok=True)

    # normalise column names
    df_gt = df_gt.rename(columns={"frame_no": "frame_no",
                                   "component": "component"})

    merged = pd.merge(df_results, df_gt,
                      on=["frame_no", "component"], how="left")
    valid  = merged.dropna(subset=["component_anomaly"])

    y_true = (valid["component_anomaly"] == "ANOMALOUS").astype(int).values
    y_pred = (valid["status"]            == "UNEXPECTED").astype(int).values

    rows = []
    for area in safety_areas:
        sub = valid[valid["component"] == area]
        if sub.empty:
            continue
        yt = (sub["component_anomaly"] == "ANOMALOUS").astype(int).values
        yp = (sub["status"]            == "UNEXPECTED").astype(int).values
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        m  = _binary_cm_metrics(cm)
        m["component"] = area
        rows.append(m)

        # confusion matrix figure
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm.astype(float))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["NORMAL", "ANOMALOUS"])
        ax.set_yticklabels(["NORMAL", "ANOMALOUS"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{area}")
        plt.tight_layout()
        fig.savefig(os.path.join(eval_dir, f"confmat_{area}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(eval_dir, "metrics_by_area.csv"), index=False)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    display_cols = ["component", "TP", "TN", "FP", "FN",
                    "accuracy", "precision", "recall", "f1"]
    print(metrics_df[[c for c in display_cols if c in metrics_df.columns]]
          .to_string(index=False))
    print("=" * 70)
    return metrics_df


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(safety_areas, models_dict, thresholds,
                  component_transforms, source,
                  args, device, out_dir):
    """
    Iterate over all frames in *source* and score each safety area.
    Returns (df_component_level, df_frame_level).
    """
    csv_dir = os.path.join(out_dir, "csv")
    fig_dir = os.path.join(out_dir, "detections")
    os.makedirs(csv_dir, exist_ok=True)
    if args.save_figures:
        os.makedirs(fig_dir, exist_ok=True)

    mask_bools = {area: component_transforms[area][2] for area in safety_areas}

    rows_component = []
    rows_frame     = []
    n_frames       = len(source) if hasattr(source, "__len__") else "?"

    pbar = tqdm(source, total=n_frames if isinstance(n_frames, int) else None,
                desc="Inference")

    for frame_t, idx, frame_path in pbar:
        if _STOP:
            break

        t0 = datetime.now()

        area_results = infer_frame(
            frame_t, safety_areas, models_dict, thresholds,
            component_transforms, args, device, args.data_source
        )

        # ── Per-component CSV rows ────────────────────────────────────────
        for area, r in area_results.items():
            rows_component.append({
                "frame_no":  idx,
                "component": area,
                "score":     r["score"],
                "norm_score":r["norm_score"],
                "threshold": r["threshold"],
                "status":    r["status"],
            })

        # ── Frame-level: anomalous if ANY component flagged ───────────────
        is_anom_frame = any(r["is_anomalous"] for r in area_results.values())
        rows_frame.append({
            "frame_no": idx,
            "status":   "ANOMALOUS" if is_anom_frame else "normal",
            "elapsed_ms": int((datetime.now() - t0).total_seconds() * 1000),
            "path":      frame_path or "",
        })

        # ── Optional figure ───────────────────────────────────────────────
        if args.save_figures and args.data_source != "preprocessed":
            save_detection_figure(
                frame_t, area_results, safety_areas, thresholds,
                mask_bools, fig_dir, idx, dpi=args.dpi
            )

        pbar.set_postfix({
            area: f"{'ANOM' if r['is_anomalous'] else 'ok'} {r['norm_score']:.2f}"
            for area, r in area_results.items()
        })

    df_comp  = pd.DataFrame(rows_component)
    df_frame = pd.DataFrame(rows_frame)

    df_comp .to_csv(os.path.join(csv_dir, "component_level.csv"),  index=False)
    df_frame.to_csv(os.path.join(csv_dir, "frame_level.csv"),      index=False)
    print(f"[save] CSVs → {csv_dir}")

    return df_comp, df_frame


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="VAE-GAN anomaly detection inference.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Data source ───────────────────────────────────────────────────────
    p.add_argument("--data_source", default="preprocessed",
                   choices=["preprocessed", "raw", "video", "ipcam"],
                   help=(
                       "preprocessed : ImageFolder with pre-cropped images\n"
                       "raw          : Full-resolution images, masked on-the-fly\n"
                       "video        : Pre-recorded video file\n"
                       "ipcam        : Live IP-camera RTSP/HTTP stream"
                   ))
    p.add_argument("--input_dir",   
                #    default=r'D:\DS\VeleriaLab\V6\fronttop\train_processed',
                   default=r'D:\DS\VeleriaLab\V6\fronttop\test_processed\unexpected_person',
                   help="Root dir for preprocessed / raw sources.")
    p.add_argument("--input_video", default=None,
                   help="Path to video file (data_source=video).")
    p.add_argument("--camera_url",  default=None,
                   help="RTSP/HTTP URL (data_source=ipcam).")
    p.add_argument("--max_frames",  default=None, type=int,
                   help="Stop after this many frames (video / ipcam).")

    # ── Safety areas ──────────────────────────────────────────────────────
    p.add_argument("--safety_area", default="RoboArm",
                   help="Safety area(s) to run. Pass 'ALL' for all areas.")

    # ── Model / paths ─────────────────────────────────────────────────────
    p.add_argument("--dataset_version",  default="V6")
    p.add_argument("--dataset_type",     default="fronttop")
    p.add_argument("--mask_image_name",  default=3015, type=int)
    p.add_argument("--latent_dims",      default=64,   type=int)
    p.add_argument("--exp_type",         default="E3")
    p.add_argument("--save_path_type",   default="cloud",
                   choices=["cloud", "local"])
    p.add_argument("--threshold_dir",    default=None,
                   help="Directory containing threshold JSON files "
                        "(default: scripts/results/training/threshold).")
    p.add_argument("--checkpoints",
                   default="scripts/dm_checkpoints/checkpoints_32", 
                   choices=["scripts/dm_checkpoints/checkpoints_32", 
                            "scripts/dm_checkpoints/checkpoints_33"])
    # ── Anomaly score params ──────────────────────────────────────────────
    p.add_argument("--offset",   default=1,   type=int)
    p.add_argument("--sigma",    default=1.0, type=float)
    p.add_argument("--quantile", default=1.0, type=float)

    # ── Output ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",   default=None,
                   help="Where to write results. "
                        "Default: scripts/results/inference/<timestamp>")
    p.add_argument("--run_name",     default=None,
                   help="Sub-folder name inside output_dir.")
    p.add_argument("--save_figures", action="store_true", default=False,
                   help="Save per-frame detection PNG figures.")
    p.add_argument("--dpi",          default=150, type=int)

    # ── Evaluation (optional) ─────────────────────────────────────────────
    p.add_argument("--gt_csv", default='scripts/data/annotation_unexpected_person.csv',
                   help="Path to ground-truth annotation CSV "
                        "(columns: frame_no, component, component_anomaly).\n"
                        "If provided, metrics and confusion matrices are computed.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
            
    # ── Safety areas ──────────────────────────────────────────────────────
    if args.safety_area.upper() == "ALL":
        safety_areas = ALL_SAFETY_AREAS
    else:
        safety_areas = [args.safety_area]
    print(f"[areas] {safety_areas}")

    # ── Params / Paths ────────────────────────────────────────────────────
    params, paths = utc.get_params_paths()
    paths         = utc.get_paths(paths, verbose=False)

    params.latent_dims    = args.latent_dims
    params.exp_type       = args.exp_type
    params.subgroup_mask  = "mask"
    params.subgroup = safety_areas[0]
    paths, params = utc.get_dataset_version(
        paths, params,
        dataset_version = args.dataset_version,
        dataset_type    = args.dataset_type,
        mask_image_name = args.mask_image_name,
        subgroup        = safety_areas[0],
        verbose         = False,
    )
    params = utc.get_parameters_by_experiment(params, verbose=False)

    paths.path_codes_cloud = paths.path_codes
    paths.path_codes_main  = os.path.join(paths.path_codes, "scripts")
    paths.path_results_cloud = os.path.join(paths.path_codes_cloud, 'scripts/results')
    os.makedirs(paths.path_codes_main, exist_ok=True)
    # paths.path_models      = os.path.join(paths.path_codes_main, args.checkpoints)
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)

    # ── Threshold directory ───────────────────────────────────────────────
    if args.threshold_dir is None:
        args.threshold_dir = os.path.join(
            paths.path_codes_main, "results", "training", "threshold"
        )

    # ── Output directory ──────────────────────────────────────────────────
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join(
        args.output_dir or os.path.join(paths.path_codes_main, "results", "inference"),
        run_name,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[output] {out_dir}")

    # ── Load models + thresholds ──────────────────────────────────────────
    models_dict, thresholds = load_models(
        safety_areas, params, paths, args, device
    )

    # ── Build component transforms (not needed for preprocessed source) ───
    component_transforms = {}
    if args.data_source != "preprocessed":
        for area in safety_areas:
            params.subgroup = area
            component_transforms[area] = build_component_transform(
                area, paths, params, args.mask_image_name
            )
    else:
        # preprocessed: dummy cropper with identity crop region
        class _NoCrop:
            x1, y1, x2, y2 = 0, 0, 128, 128
        for area in safety_areas:
            _, _, _, mask_bool_dummy = 0, 0, 0, np.ones((128, 128), dtype=bool)
            component_transforms[area] = (None, _NoCrop(), mask_bool_dummy)

    # ── Build data source ─────────────────────────────────────────────────
    source = build_source(args)

    # ── Run inference ─────────────────────────────────────────────────────
    utc.get_time(suff="start")
    df_comp, df_frame = run_inference(
        safety_areas, models_dict, thresholds,
        component_transforms, source,
        args, device, out_dir
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("INFERENCE SUMMARY")
    print(f"{'='*70}")
    print(f"Frames processed : {df_frame.shape[0]}")
    print(f"ANOMALOUS frames : {(df_frame.status == 'ANOMALOUS').sum()}")
    print(f"Normal frames    : {(df_frame.status == 'normal').sum()}")
    for area in safety_areas:
        sub = df_comp[df_comp.component == area]
        n_anom = (sub.status == "UNEXPECTED").sum()
        print(f"  {area:<12} UNEXPECTED={n_anom}/{len(sub)}"
              f"  tau={thresholds[area]:.4f}"
              f"  mean_score={sub.score.mean():.4f}")
    print(f"{'='*70}")

    # ── Optional evaluation vs ground-truth ───────────────────────────────
    if args.gt_csv:
        if not os.path.exists(args.gt_csv):
            print(f"[warn] GT CSV not found: {args.gt_csv}  — skipping evaluation.")
        else:
            df_gt     = pd.read_csv(args.gt_csv)
            eval_dir  = os.path.join(out_dir, "evaluation")
            evaluate(df_comp, df_gt, eval_dir, safety_areas)

    utc.get_time(suff="end")
    print(f"[done] Results in: {out_dir}")


if __name__ == "__main__":
    main()
