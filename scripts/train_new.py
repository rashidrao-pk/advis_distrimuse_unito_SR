"""
train_CAD.py
------------
VAE-GAN training script for Collaborative Anomaly Detection (CAD).
Converted from notebook: N4_train_CAD_updated_model_train_s1.ipynb

Architecture : Encoder + Decoder (VAE) + Discriminator (GAN)
Loss         : Reconstruction (MSE) + KL divergence + Adversarial (BCEWithLogits)
"""

import os
import csv
import json
import math
import argparse
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

import yaml

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import utils as ut
import utils_model as utmc
from utils_model import Encoder, Decoder, Discriminator

# ---------------------------------------------------------------------------
# Global stop flag — replaces the ipywidgets checkbox
# ---------------------------------------------------------------------------
_STOP_TRAINING = False

def _handle_sigint(sig, frame):
    global _STOP_TRAINING
    print("\n[INFO] SIGINT received — stopping after current epoch.")
    _STOP_TRAINING = True

signal.signal(signal.SIGINT, _handle_sigint)


# ---------------------------------------------------------------------------
# Dataset helpers (inlined from notebook Cell 15)
# ---------------------------------------------------------------------------

class DatasetFromPaths(Dataset):
    """Dataset built from an explicit list of (path, class_idx) samples."""

    def __init__(self, samples, class_to_idx, transform=None):
        self.samples     = samples
        self.imgs        = samples
        self.targets     = [s[1] for s in samples]
        self.class_to_idx = class_to_idx
        self.classes     = [None] * len(class_to_idx)
        for cls_name, idx in class_to_idx.items():
            self.classes[idx] = cls_name
        self.transform   = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_transforms(augmentation_type: str = "min"):
    if augmentation_type == "min":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif augmentation_type == "custom":
        transform_train = transforms.Compose([
            transforms.RandomAffine(
                degrees=0.01,
                translate=(0.01, 0.01),
                shear=0.1,
                scale=(0.99, 1.0),
                fill=(0, 0, 0),
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        raise ValueError(f"augmentation_type='{augmentation_type}' is not supported.")

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform_train, transform_val


def build_video_4to1_split(
    root_dir,
    split_save_path,
    val_every: int = 5,
    val_offset: int = 4,
    split_strategy: str = "scenario",
    verbose: bool = True,
):
    """Build a deterministic train/val split and save it as JSON."""
    print(' [-] DEBUGG MODE --')
    print('LOADING DATA FROM --', root_dir)
    print(' [-] DEBUGG MODE --')

    base_dataset = datasets.ImageFolder(root=root_dir)
    samples      = list(base_dataset.samples)

    samples_sorted = sorted(enumerate(samples), key=lambda x: x[1][0])

    train_indices, val_indices = [], []
    validation_scenarios = []
    if split_strategy == "scenario":
        scenario_pattern = re.compile(r"(?:^|[/_])s-(\d+_\d+)(?:_|$)")
        scenarios = {}
        for old_idx, (path, _) in enumerate(samples):
            match = scenario_pattern.search(path.replace(os.sep, "/"))
            if match:
                scenarios.setdefault(match.group(1), []).append(old_idx)
        if len(scenarios) >= 2:
            scenario_ids = sorted(
                scenarios, key=lambda value: tuple(int(part) for part in value.split("_"))
            )
            validation_scenarios = [
                sid for index, sid in enumerate(scenario_ids)
                if index % val_every == val_offset
            ]
            if not validation_scenarios:
                validation_scenarios = [scenario_ids[-1]]
            if len(validation_scenarios) == len(scenario_ids):
                validation_scenarios = [scenario_ids[-1]]
            validation_set = set(validation_scenarios)
            assigned_indices = set()
            for sid, indices in scenarios.items():
                (val_indices if sid in validation_set else train_indices).extend(indices)
                assigned_indices.update(indices)
            train_indices.extend(
                index for index in range(len(samples)) if index not in assigned_indices
            )
        else:
            print("[split] Fewer than two scenario IDs found; using contiguous holdout.")
            ordered = [old_idx for old_idx, _ in samples_sorted]
            val_count = max(1, len(ordered) // val_every)
            train_indices, val_indices = ordered[:-val_count], ordered[-val_count:]
    else:
        for sorted_idx, (old_idx, _) in enumerate(samples_sorted):
            if sorted_idx % val_every == val_offset:
                val_indices.append(old_idx)
            else:
                train_indices.append(old_idx)

    split_info = {
        "root_dir":     str(root_dir),
        "rule":         {
            "type":        split_strategy,
            "val_every":   int(val_every),
            "val_offset":  int(val_offset),
            "description": (
                "complete scenario holdout"
                if split_strategy == "scenario"
                else f"sorted_index % {val_every} == {val_offset} -> val, else train"
            ),
            "validation_scenarios": validation_scenarios,
        },
        "num_total":     len(samples),
        "num_train":     len(train_indices),
        "num_val":       len(val_indices),
        "train_indices": train_indices,
        "val_indices":   val_indices,
        "train_paths":   [samples[i][0] for i in train_indices],
        "val_paths":     [samples[i][0] for i in val_indices],
        "class_to_idx":  base_dataset.class_to_idx,
    }

    split_save_path = Path(split_save_path)
    split_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_save_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    if verbose:
        print(f"[split] Saved to: {split_save_path}")
        print(f"[split] Total: {len(samples)} | Train: {len(train_indices)} | Val: {len(val_indices)}")

    return split_info


def prepare_or_load_video_split(
    train_dir_processed_subgroup,
    split_save_path,
    val_every: int = 5,
    val_offset: int = 4,
    split_strategy: str = "scenario",
    force_rebuild: bool = False,
    verbose: bool = True,
):
    split_save_path = Path(split_save_path)
    if force_rebuild or not split_save_path.exists():
        if verbose:
            print("[split] Building new split...")
        return build_video_4to1_split(
            root_dir=train_dir_processed_subgroup,
            split_save_path=split_save_path,
            val_every=val_every,
            val_offset=val_offset,
            split_strategy=split_strategy,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"[split] Loading existing split from: {split_save_path}")
        with open(split_save_path, "r", encoding="utf-8") as f:
            return json.load(f)


def get_data_loaders_from_preprocessed_with_saved_split(
    train_dir_processed_subgroup,
    split_save_path,
    augmentation_type: str = "min",
    batch_size: int = 32,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    drop_last_train: bool = True,
    drop_last_val: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    verbose: bool = False,
):
    """Return (train_loader, val_loader, train_dataset, val_dataset, split_info)."""
    if verbose:
        print(f"[data] Building loaders | aug={augmentation_type} | batch={batch_size}")

    base_dataset = datasets.ImageFolder(root=train_dir_processed_subgroup)
    samples      = list(base_dataset.samples)

    with open(split_save_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    transform_train, transform_val = get_transforms(augmentation_type)

    train_dataset = DatasetFromPaths(
        samples      = [samples[i] for i in split_info["train_indices"]],
        class_to_idx = base_dataset.class_to_idx,
        transform    = transform_train,
    )
    val_dataset = DatasetFromPaths(
        samples      = [samples[i] for i in split_info["val_indices"]],
        class_to_idx = base_dataset.class_to_idx,
        transform    = transform_val,
    )

    _pw = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size        = batch_size,
        shuffle           = shuffle_train,
        num_workers       = num_workers,
        pin_memory        = pin_memory,
        persistent_workers= _pw,
        drop_last         = drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size        = batch_size,
        shuffle           = shuffle_val,
        num_workers       = num_workers,
        pin_memory        = pin_memory,
        persistent_workers= _pw,
        drop_last         = drop_last_val,
    )

    if verbose:
        print(f"[data] train={len(train_dataset)} | val={len(val_dataset)}")

    return train_loader, val_loader, train_dataset, val_dataset, split_info


def format_timedelta_human(td):
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        total_seconds = 0

    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


@torch.no_grad()
def collect_one_class_scores(loader, Enc, Dec, device, max_batches=None):
    """Return deterministic per-image reconstruction and KL scores."""
    was_training_enc, was_training_dec = Enc.training, Dec.training
    Enc.eval(); Dec.eval()
    reconstruction_scores, kl_scores = [], []
    for batch_index, (images, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        z_mean, z_logvar = Enc(images)
        reconstructed = Dec(z_mean)
        reconstruction = torch.mean((reconstructed - images) ** 2, dim=(1, 2, 3))
        kl = -0.5 * torch.mean(
            1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1
        )
        reconstruction_scores.extend(reconstruction.cpu().tolist())
        kl_scores.extend(kl.cpu().tolist())
    Enc.train(was_training_enc); Dec.train(was_training_dec)
    return np.asarray(reconstruction_scores), np.asarray(kl_scores)


def evaluate_one_class_validation(val_loader, anomaly_loader, Enc, Dec, args, device):
    """Compute valid normal-only metrics and optional two-class anomaly metrics."""
    normal_scores, normal_kl = collect_one_class_scores(
        val_loader, Enc, Dec, device, args.validation_max_batches
    )
    if len(normal_scores) < 2:
        raise RuntimeError("At least two validation images are required")

    # With one-class data, use one half to estimate the normal boundary and the
    # other half to measure how often unseen normal images exceed it.
    split = max(1, len(normal_scores) // 2)
    calibration = normal_scores[:split]
    evaluation = normal_scores[split:] if split < len(normal_scores) else normal_scores
    threshold = float(np.percentile(calibration, args.normal_threshold_percentile))
    normal_predictions = evaluation > threshold
    mean = float(normal_scores.mean())
    metrics = {
        "val_recon_mean": mean,
        "val_recon_median": float(np.median(normal_scores)),
        "val_recon_std": float(normal_scores.std()),
        "val_recon_cv": float(normal_scores.std() / mean) if mean > 0 else 0.0,
        "val_recon_p95": float(np.percentile(normal_scores, 95)),
        "val_recon_p99": float(np.percentile(normal_scores, 99)),
        "val_kl_mean": float(normal_kl.mean()),
        "normal_threshold": threshold,
        "val_normal_fpr": float(normal_predictions.mean()),
        "val_auroc": math.nan,
        "val_auprc": math.nan,
        "val_precision": math.nan,
        "val_recall": math.nan,
        "val_f1": math.nan,
    }

    if anomaly_loader is not None:
        anomaly_scores, _ = collect_one_class_scores(
            anomaly_loader, Enc, Dec, device, args.validation_max_batches
        )
        labels = np.concatenate([
            np.zeros(len(evaluation), dtype=np.int64),
            np.ones(len(anomaly_scores), dtype=np.int64),
        ])
        scores = np.concatenate([evaluation, anomaly_scores])
        predictions = (scores > threshold).astype(np.int64)
        metrics.update({
            "val_auroc": float(roc_auc_score(labels, scores)),
            "val_auprc": float(average_precision_score(labels, scores)),
            "val_precision": float(precision_score(labels, predictions, zero_division=0)),
            "val_recall": float(recall_score(labels, predictions, zero_division=0)),
            "val_f1": float(f1_score(labels, predictions, zero_division=0)),
        })
    return metrics


def save_validation_history(loss_history, params, paths):
    """Save one-class validation history as CSV and a compact diagnostic plot."""
    output_dir = Path(paths.path_training_curves)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in loss_history if "val_recon_mean" in row]
    if not rows:
        return
    csv_path = output_dir / f"validation_metrics_{params.subgroup}.csv"
    fields = ["epoch"] + sorted({key for row in rows for key in row if key.startswith("val_") or key == "normal_threshold"})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch, row in enumerate(rows, start=1):
            writer.writerow({"epoch": epoch, **{field: row.get(field, math.nan) for field in fields[1:]}})

    epochs = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axes[0, 0].plot(epochs, [row["val_recon_mean"] for row in rows], label="Mean")
    axes[0, 0].plot(epochs, [row["val_recon_p95"] for row in rows], label="P95")
    axes[0, 0].plot(epochs, [row["val_recon_p99"] for row in rows], label="P99")
    axes[0, 0].set_title("Normal validation reconstruction scores")
    axes[0, 0].set_ylabel("Reconstruction error")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["val_recon_std"] for row in rows], label="Std. deviation")
    axes[0, 1].plot(epochs, [row["val_recon_cv"] for row in rows], label="Coefficient of variation")
    axes[0, 1].set_title("Normal validation-score variability")
    axes[0, 1].legend()

    threshold_axis = axes[1, 0]
    fpr_axis = threshold_axis.twinx()
    threshold_axis.plot(epochs, [row["normal_threshold"] for row in rows],
                            color="tab:orange", label="Normal threshold")
    fpr_axis.plot(epochs, [row["val_normal_fpr"] for row in rows],
                     color="tab:red", label="Normal FPR")
    threshold_axis.set_title("One-class operating point")
    threshold_axis.set_ylabel("Threshold", color="tab:orange")
    fpr_axis.set_ylabel("False-positive rate", color="tab:red")
    fpr_axis.set_ylim(0, 1)
    lines = threshold_axis.lines + fpr_axis.lines
    threshold_axis.legend(lines, [line.get_label() for line in lines])

    supervised_plotted = False
    for key, label in (("val_auroc", "AUROC"), ("val_auprc", "AUPRC"),
                       ("val_precision", "Precision"), ("val_recall", "Recall"),
                       ("val_f1", "F1")):
        values = np.asarray([row.get(key, math.nan) for row in rows], dtype=float)
        if np.isfinite(values).any():
            axes[1, 1].plot(epochs, values, label=label)
            supervised_plotted = True
    axes[1, 1].set_title("Optional labeled-anomaly metrics")
    axes[1, 1].set_ylim(0, 1.02)
    if supervised_plotted:
        axes[1, 1].legend()
    else:
        axes[1, 1].text(
            0.5, 0.5, "No anomalous validation set\n(one-class mode)",
            ha="center", va="center", transform=axes[1, 1].transAxes,
        )

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    fig.suptitle(f"Validation diagnostics — {params.subgroup}", fontsize=15)
    fig.savefig(
        output_dir / f"validation_metrics_{params.subgroup}.png",
        dpi=160, bbox_inches="tight",
    )
    plt.close(fig)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_loader,
    val_loader,
    anomaly_loader,
    Enc, Dec, Dis,
    optEncDec, optDis,
    reconstruction_loss_fn,
    adversarial_loss_fn,
    loss_history,
    args,
    params, paths, suffix,
    device,
    verbose_print: bool = True,
    verbose_level: int  = 1,
    model_save_interval: int = 10,
    data_train_fx       = None,
    save_figures: bool  = False,
):
    global _STOP_TRAINING

    if args.verbose_level > 0:
        print("TRAINING STARTED")
        print("-" * 100)
        print(f"Total Images in Training set --> {len(train_loader.dataset)}")

    beta_kl  = params.beta_kl
    beta_gan = params.beta_gan

    start_time  = datetime.now()
    log_messages = ut.create_log_file(params, paths, start_time, verbose=True and args.verbose_level > 0)
    epoch_time_estimate_printed = False
    iterator = tqdm(
        range(params.epochs),
        initial     = len(loss_history),
        total       = params.epochs,
        desc        = "Training Epochs",
        position    = 0,
        leave       = False,
    )

    for iter_num_id, iter_num in enumerate(iterator):
        epoch_start_time = datetime.now()
        # ── early-timing estimate after first batch ──────────────────────────
        if iter_num_id == 1:
            pending_epochs    = params.epochs - len(loss_history)
            start_time_first  = datetime.now()

        # ── stop conditions ──────────────────────────────────────────────────
        if _STOP_TRAINING:
            print("[INFO] Stop flag set — exiting training loop.")
            break
        if len(loss_history) >= params.epochs:
            break

        epoch_number = len(loss_history) + 1
        use_dis_for_vae_training = (
            not args.disable_gan and epoch_number > args.gan_warmup_epochs
        )
        train_dis = (
            use_dis_for_vae_training
            and (epoch_number - args.gan_warmup_epochs - 1)
            % args.discriminator_update_every == 0
        )

        epoch_loss = {
            "recon_loss":      0.0,
            "kl_loss":         0.0,
            "gan_loss":        0.0,
            "beta_kl_loss":    0.0,
            "beta_gan_loss":   0.0,
            "vae_loss":        0.0,
            "disc_loss":       0.0,
            "annealing_lambda":0.0,
        }
        dis_preds, dis_labels = [], []

        batch_iterator = tqdm(train_loader, desc="Training Batches", position=1, leave=False)

        for real_images, _ in batch_iterator:
            batch_size      = real_images.size(0)
            real_images_dev = real_images.to(device)

            # ── Step 1: Encoder + Decoder ────────────────────────────────────
            optEncDec.zero_grad()

            z_mean, z_logvar     = Enc(real_images_dev)
            std                  = torch.exp(0.5 * z_logvar)
            z                    = z_mean + torch.randn_like(std) * std
            reconstructed_images = Dec(z)

            recon_loss    = reconstruction_loss_fn(reconstructed_images, real_images_dev)
            kl_divergence = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

            if use_dis_for_vae_training:
                fake_logits = Dis(reconstructed_images)
                gan_loss    = adversarial_loss_fn(fake_logits, torch.ones_like(fake_logits))
            else:
                gan_loss = torch.tensor([0.0], device=device)

            annealing_lambda = min(1.0, epoch_number / max(1, args.kl_anneal_epochs))

            lossEncDec = (
                recon_loss
                + annealing_lambda * beta_kl  * kl_divergence
                + annealing_lambda * beta_gan * gan_loss
            )
            lossEncDec.backward()
            optEncDec.step()

            # ── Step 2: Discriminator ────────────────────────────────────────
            if train_dis:
                optDis.zero_grad()

                real_logits   = Dis(real_images_dev)
                real_targets = torch.full_like(real_logits, args.real_label_smoothing)
                lossDis_real  = adversarial_loss_fn(real_logits, real_targets)

                fake_logits   = Dis(reconstructed_images.detach())
                lossDis_fake  = adversarial_loss_fn(fake_logits, torch.zeros_like(fake_logits))

                lossDis = (lossDis_real + lossDis_fake) / 2
                lossDis.backward()
                optDis.step()

                real_preds = (torch.sigmoid(real_logits).detach().cpu().numpy() > 0.5).astype(int)
                fake_preds = (torch.sigmoid(fake_logits).detach().cpu().numpy() > 0.5).astype(int)
                dis_preds.extend(real_preds.flatten());  dis_labels.extend(np.ones(batch_size))
                dis_preds.extend(fake_preds.flatten());  dis_labels.extend(np.zeros(batch_size))
            else:
                lossDis = torch.tensor([math.nan], device=device)

            # ── Accumulate batch losses ──────────────────────────────────────
            epoch_loss["recon_loss"]       += recon_loss.item()
            epoch_loss["kl_loss"]          += kl_divergence.item()
            epoch_loss["gan_loss"]         += gan_loss.item()
            epoch_loss["beta_kl_loss"]     += annealing_lambda * beta_kl  * kl_divergence.item()
            epoch_loss["beta_gan_loss"]    += annealing_lambda * beta_gan * gan_loss.item()
            epoch_loss["vae_loss"]         += lossEncDec.item()
            epoch_loss["disc_loss"]        += lossDis.item()
            epoch_loss["annealing_lambda"] += annealing_lambda

        # ── Per-epoch averaging ──────────────────────────────────────────────
        n_batches = len(train_loader)
        for key in epoch_loss:
            epoch_loss[key] /= n_batches

        epoch_loss["dis_acc"] = (
            accuracy_score(dis_labels, dis_preds) if dis_labels else math.nan
        )
        epoch_loss["dis_F1"] = (
            f1_score(dis_labels, dis_preds) if dis_labels else math.nan
        )
        validation_metrics = evaluate_one_class_validation(
            val_loader, anomaly_loader, Enc, Dec, args, device
        )
        epoch_loss.update(validation_metrics)
        loss_history.append(epoch_loss)
        save_validation_history(loss_history, params, paths)
        if args.verbose_level > 0:
            print(
                "[validation] "
                f"recon={validation_metrics['val_recon_mean']:.6f} "
                f"p95={validation_metrics['val_recon_p95']:.6f} "
                f"variability(CV)={validation_metrics['val_recon_cv']:.3f} "
                f"normal-FPR={validation_metrics['val_normal_fpr']:.2%}"
            )
            if math.isfinite(validation_metrics["val_auroc"]):
                print(
                    f"[validation/anomaly] AUROC={validation_metrics['val_auroc']:.4f} "
                    f"AUPRC={validation_metrics['val_auprc']:.4f} "
                    f"F1={validation_metrics['val_f1']:.4f}"
                )

        # ── Visualisation / monitoring ───────────────────────────────────────
        if save_figures:
            colormap_anomaly_map = ut.get_colormap()  # noqa: F841
            ut.plot_images(
                real_images, reconstructed_images, len(loss_history), paths,
                plot_anomaly_scores=True, save_fig=True, interval=args.save_fig_interval, destroy_fig=True,
            )

            if data_train_fx is not None:
                recon_train_fx = utmc.get_reconstructed(Enc, Dec, data_train_fx, device=device)
                z_random       = torch.randn((params.batch_size, params.latent_dims), device=device)
                fake_images    = Dec(z_random)
                ut.plot_images_tracking(
                    real_images, reconstructed_images,
                    data_train_fx, recon_train_fx,
                    torch.zeros_like(data_train_fx), fake_images,
                    iter_num, paths.path_results_fix, ttl="train",
                    plot_anomaly_scores=False, destroy_fig=True,
                    save_fig=True,
                    interval=5,
                )

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if (iter_num==0) or ((iter_num + 1) % model_save_interval == 0):
            # utmc.save_model(Enc, Dec, Dis, optEncDec, optDis, paths, loss_history, suffix)
            utmc.save_model(Enc=Enc,Dec=Dec,D=Dis,
                    optEncDec=optEncDec,
                    optD=optDis,
                    loss_history=loss_history,
                    path_models=paths.path_models,
                    suffix=f"{params.subgroup}_{params.latent_dims}",
                    epoch=params.epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    params={"latent_dim": params.latent_dims,
                            "lr_encdec": params.learning_rate_enc_dec,
                            "lr_d": params.learning_rate_dis,
                            "image_size": (params.input_shape[1],params.input_shape[-1]) ,
                            "beta_gan": params.beta_gan,
                            "beta_kl": params.beta_kl,
                            },
                    augmentation=train_loader.dataset.transform,
                    dataset_name= f'{args.dataset_source}_{args.dataset_version}_{args.dataset_cam_type}', #"MVTec_hazelnut",
                    train_dir=paths.train_dir_processed_subgroup,   
                    notes="VAE-GAN trained on normal images only",
                    verbose=True and args.verbose_level > 1)

        # ── ETA after first real epoch ───────────────────────────────────────
                # ── ETA after first completed epoch ───────────────────────────────
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time

        if args.estimate_time and not epoch_time_estimate_printed:
            completed_epochs = len(loss_history)
            remaining_epochs = max(params.epochs - completed_epochs, 0)
            total_epochs_target = params.epochs

            estimated_remaining = epoch_duration * remaining_epochs
            estimated_total = epoch_duration * total_epochs_target

            timing_msg = (
                f"[timing] 1 epoch took {format_timedelta_human(epoch_duration)} "
                f"| {remaining_epochs} remaining epochs might take {format_timedelta_human(estimated_remaining)} "
                f"| {total_epochs_target} total epochs might take {format_timedelta_human(estimated_total)}"
            )

            print(timing_msg)
            log_messages += timing_msg + "\n"
            epoch_time_estimate_printed = True

        # ── Console log ──────────────────────────────────────────────────────
        table_msg = (
            f"| Epoch: [{len(loss_history):>5}] "
            f"| LossEncDec: {lossEncDec.item():<10.5f} "
            f"| LossDis: {lossDis.item():<10.5f} |\n"
        )
        if verbose_print and args.verbose_level > 0:
            print(table_msg, end="")
        log_messages += table_msg

        # ── Loss curves — always saved to results/training ───────────────────
        
        ut.plot_loss_sep(loss_history, params, paths)
        # ut.plot_losses(loss_history, params, paths)

    return loss_history,log_messages


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# All known safety areas — used when --safety_area ALL is passed
ALL_SAFETY_AREAS = ["PLeft", "PRight","RoboArm", "ConvBelt"]


def _resolve_config_path(value, config_path):
    """Resolve a data path from YAML, relative to the YAML file when needed."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def load_data_config(config_file):
    """Load dataset paths, deriving the train folder from dataset_base."""
    config_path = Path(config_file).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a 'data' mapping: {config_path}")

    dataset_base = _resolve_config_path(data.get("dataset_base"), config_path)
    training_value = data.get("training") or data.get("train")
    if training_value is not None:
        training = _resolve_config_path(training_value, config_path)
    elif dataset_base is not None:
        training = str((Path(dataset_base) / "train").resolve())
    else:
        training = None

    return {
        "dataset_base": dataset_base,
        "training": training,
        "masks": _resolve_config_path(data.get("masks"), config_path),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE-GAN for CAD anomaly detection")
    p.add_argument("--safety_area",         default="RoboArm",
                   help="Safety area (subgroup) to train. Pass 'ALL' to train every area sequentially.")
    
    p.add_argument("--dataset_source",     default="SR",       help="Dataset version tag")
    p.add_argument("--dataset_version",     default="V6",       help="Dataset version tag")
    p.add_argument("--dataset_cam_type",    default="refined", help="Camera / dataset type")
    p.add_argument(
        "--config", type=Path, default=Path("configs/cf_dataset_mac.yaml"),
        help=("YAML dataset config. Training uses data.training/data.train when set, "
              "otherwise <data.dataset_base>/train."),
    )
    p.add_argument("--dataset_base", help="Dataset base directory (overrides data.dataset_base in --config)")
    p.add_argument("--training_dir", help="Training directory (overrides data.training in --config)")
    p.add_argument("--masks_dir", help="Masks directory (overrides data.masks in --config)")
    p.add_argument("--model_path",          default="scripts/models", help="Model Save Path")
    p.add_argument("--mask_image_name",     default=3015,   type=int)
    p.add_argument("--epochs",              default=200,   type=int)
    p.add_argument("--batch_size",          default=64,     type=int)
    p.add_argument("--latent_dims",         default=64,     type=int)
    p.add_argument("--exp_type",            default="E3", help="Experiment type key for ut.get_parameters_by_experiment")
    p.add_argument("--augmentation_type",   default="custom", choices=["min", "custom"])
    p.add_argument(
        "--gan_warmup_epochs", default=5, type=int,
        help="Train the VAE without adversarial loss for the first N epochs.",
    )
    p.add_argument(
        "--kl_anneal_epochs", default=20, type=int,
        help="Linearly increase KL/GAN weights to full strength over N epochs.",
    )
    p.add_argument(
        "--discriminator_update_every", default=2, type=int,
        help="Update the discriminator once every N epochs (helps prevent saturation).",
    )
    p.add_argument(
        "--real_label_smoothing", default=0.9, type=float,
        help="Smoothed discriminator target for real images.",
    )
    p.add_argument(
        "--disable_gan", action="store_true",
        help="Train a VAE-only baseline without adversarial loss.",
    )
    p.add_argument("--num_workers",         default=4,      type=int)
    p.add_argument("--pin_memory",          default=True,  type=bool)
    p.add_argument("--val_every",           default=5,      type=int, help="1-in-N frames goes to validation")
    p.add_argument("--val_offset",          default=4,      type=int)
    p.add_argument(
        "--split_strategy", default="scenario", choices=["scenario", "frame"],
        help="Hold out complete scenarios (recommended) or use the legacy per-frame split.",
    )
    p.add_argument(
        "--normal_threshold_percentile", default=99.0, type=float,
        help="Percentile of normal validation-calibration scores used as anomaly threshold.",
    )
    p.add_argument(
        "--validation_max_batches", default=None, type=int,
        help="Optional limit for validation batches per epoch; default evaluates all.",
    )
    p.add_argument(
        "--anomaly_validation_dir", type=Path,
        help=("Optional ImageFolder root containing labeled anomalous validation images. "
              "Without it, only statistically valid one-class metrics are reported."),
    )
    p.add_argument("--force_rebuild_split", action="store_true", help="Force rebuild the train/val split JSON")
    p.add_argument("--model_override",      action="store_true", help="Rename existing checkpoint before training")
    p.add_argument("--model_save_interval", default=5,     type=int)
    p.add_argument("--save_fig_interval", default=2,     type=int)
    p.add_argument("--verbose_level",       default=0,      type=int, choices=[0, 1, 2])
    p.add_argument("--save_path_type",      default="cloud", choices=["cloud", "local"])
    p.add_argument("--dry_run",             action="store_true", default=False)
    p.add_argument("--checkpoints",
                   default="scripts/results/models",
                   help="Checkpoint output directory (relative paths use the current working directory).")
    p.add_argument("--save_figures",        action="store_true", default=False,
                   help="Save reconstruction & tracking figures during training. "
                        "When disabled only loss curves (results/training) and model "
                        "checkpoints (results/models) are written.")
    p.add_argument("--estimate_time",action="store_true",help="Estimate training time after the first completed epoch.")
    args = p.parse_args()

    if args.discriminator_update_every < 1:
        p.error("--discriminator_update_every must be at least 1")
    if not 0.0 < args.real_label_smoothing <= 1.0:
        p.error("--real_label_smoothing must be in (0, 1]")
    if not 0.0 < args.normal_threshold_percentile < 100.0:
        p.error("--normal_threshold_percentile must be in (0, 100)")

    config_data = {}
    if args.config:
        try:
            config_data = load_data_config(args.config)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            p.error(str(exc))

    # Explicit command-line paths have priority over values from the YAML file.
    args.dataset_base = args.dataset_base or config_data.get("dataset_base")
    args.training_dir = args.training_dir or config_data.get("training")
    args.masks_dir = args.masks_dir or config_data.get("masks")
    return args

def train_one_safety_area(safety_area: str, args, device):
    """Set up and train a single safety area. Returns when done or interrupted."""
    global _STOP_TRAINING

    print(f"\n{'='*100}")
    print(f"[safety_area] Starting: {safety_area}")
    print(f"{'='*100}")

    # ── Params / Paths ────────────────────────────────────────────────────
    params, paths = ut.get_params_paths()
    paths         = ut.get_paths(paths, verbose=False)

    params.subgroup      = safety_area          # ut internals still use .subgroup
    params.epochs        = args.epochs
    params.batch_size    = args.batch_size
    params.latent_dims   = args.latent_dims
    params.exp_type      = args.exp_type

    paths, params = ut.get_dataset_version(
        paths, params,
        dataset_version  = args.dataset_version,
        dataset_type     = args.dataset_cam_type,
        mask_image_name  = args.mask_image_name,
        subgroup         = safety_area,
        verbose          = True and args.verbose_level > 1,
    )

    # Explicit paths (from CLI or YAML) override the legacy constructed path.
    # The training directory contains one ImageFolder directory per safety area.
    if args.dataset_base:
        paths.path_datasets = args.dataset_base
    if args.masks_dir:
        paths.mask_dir = args.masks_dir
    if args.training_dir:
        paths.train_dir_processed = args.training_dir
        paths.train_dir_processed_subgroup = os.path.join(args.training_dir, safety_area)

    if not os.path.isdir(paths.train_dir_processed_subgroup):
        raise FileNotFoundError(
            "Training directory for safety area "
            f"'{safety_area}' does not exist: {paths.train_dir_processed_subgroup}"
        )

    params = ut.get_parameters_by_experiment(params, verbose=True and args.verbose_level > 0)
    _ = ut.get_header(params, paths, verbose=True and args.verbose_level > 1)

    # ── Output dirs ───────────────────────────────────────────────────────
    paths.path_codes_cloud   = paths.path_codes
    paths.path_codes_main    = os.path.join(paths.path_codes, 'scripts')
    paths.path_codes_local   = os.path.join(paths.path_results_local, 'scripts')
    paths.path_results_cloud = os.path.join(paths.path_codes_cloud,'results', paths.dataset_version)
    paths.history_fname      = "vae_gan_train_history.csv"
    os.makedirs(paths.path_codes_main, exist_ok=True)

    # Fixed output dirs used regardless of save_figures flag
    paths.path_training_curves = os.path.join(paths.path_codes_cloud, "results", paths.dataset_version,"training")
    # paths.path_models      = os.path.join(paths.path_codes_main, args.checkpoints)
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
    os.makedirs(paths.path_training_curves, exist_ok=True)
    # os.makedirs(paths.path_models,          exist_ok=True)
    if args.verbose_level > 0:
        if args.save_figures:
            print("[save] Figure saving ENABLED  — reconstruction & tracking images will be written.")
        else:
            print("[save] Figure saving DISABLED — only loss curves and checkpoints will be written.")
            print(f"       Loss curves → {paths.path_training_curves}")
            print(f"       Checkpoints → {paths.path_models}")

    suffix, paths = ut.get_create_results_path(
        params.subgroup, params, args,paths,
        save_path_type = args.save_path_type,
        dir            = args.model_path,
        models_dir      = f'models_{args.dataset_version}',
        verbose        = True  and args.verbose_level > 1,
    )
    paths.suffix = suffix
    # ── Data split ────────────────────────────────────────────────────────
    split_save_path = os.path.join(
        paths.train_dir_processed_subgroup,
        f"split_{args.split_strategy}_4train_1val_{safety_area}.json",
    )
    prepare_or_load_video_split(
        train_dir_processed_subgroup = paths.train_dir_processed_subgroup,
        split_save_path  = split_save_path,
        val_every        = args.val_every,
        val_offset       = args.val_offset,
        split_strategy   = args.split_strategy,
        force_rebuild    = args.force_rebuild_split,
        verbose          = True  and args.verbose_level > 1,
    )

    train_loader, val_loader, train_dataset, val_dataset, split_info = \
        get_data_loaders_from_preprocessed_with_saved_split(
            train_dir_processed_subgroup = paths.train_dir_processed_subgroup,
            split_save_path   = split_save_path,
            augmentation_type = args.augmentation_type,
            batch_size        = params.batch_size,
            shuffle_train     = True,
            shuffle_val       = False,
            drop_last_train   = True,
            drop_last_val     = False,
            num_workers       = args.num_workers,
            pin_memory        = False,
            persistent_workers= False,
            verbose           = True and args.verbose_level > 1,
        )

    anomaly_loader = None
    if args.anomaly_validation_dir:
        anomaly_path = args.anomaly_validation_dir.expanduser().resolve()
        if not anomaly_path.is_dir():
            raise FileNotFoundError(
                f"Anomaly validation directory does not exist: {anomaly_path}"
            )
        _, anomaly_transform = get_transforms("min")
        anomaly_dataset = datasets.ImageFolder(
            root=str(anomaly_path), transform=anomaly_transform
        )
        anomaly_loader = DataLoader(
            anomaly_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        print(f"Anomalous validation samples: {len(anomaly_dataset)}")

    paths.train_classes     = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
    paths.class_names_train = train_dataset.classes
    if args.verbose_level >= 0:

        print(f"Samples (Train/Validation) : {len(train_dataset)} / {len(val_dataset)}")
        print(f"Data Loaded from : {paths.train_dir_processed_subgroup}")
        print('-'*80)
        print('Batch Size : {}'.format(args.batch_size))
        print('Num Workers : {}'.format(args.num_workers))
        print('Pin Memory : {}'.format(args.pin_memory))
        print('-'*80)
        print(f'Saving Figures: {"True" if args.save_figures else "False"}')
        print('-'*80)

    # ── Fixed monitoring batch ────────────────────────────────────────────
    data_train_fx, _ = next(iter(train_loader))

    # ── Models ───────────────────────────────────────────────────────────
    reconstruction_loss_fn, adversarial_loss_fn = utmc.get_loss_functions(verbose=True and args.verbose_level > 0)

    Enc = Encoder(z_size=params.latent_dims).to(device)
    Dec = Decoder(z_size=params.latent_dims).to(device)
    Dis = Discriminator().to(device)
    if args.verbose_level >= 1:
        print(f"Encoder params      : {len(list(Enc.parameters()))}")
        print(f"Decoder params      : {len(list(Dec.parameters()))}")
        print(f"Discriminator params: {len(list(Dis.parameters()))}")

    optEncDec = optim.Adam(
        list(Enc.parameters()) + list(Dec.parameters()),
        lr=params.learning_rate_enc_dec,
    )
    optDis = optim.Adam(Dis.parameters(), lr=params.learning_rate_dis)

    # ── Optional checkpoint override ─────────────────────────────────────
    if args.model_override:
        utmc.model_override(paths.path_models, suffix)

    # ── Resume from checkpoint ────────────────────────────────────────────
    # loss_history = utmc.load_model(
    #     Enc, Dec, Dis, optEncDec, optDis, paths, suffix, device=device, verbose=True and args.verbose_level > 0
    # )
    loss_history, config = utmc.load_model(Enc, Dec, Dis, optEncDec, optDis,
                                           path_models=paths.path_models,
                                           suffix=f"{params.subgroup}_{params.latent_dims}",
                                           verbose=True,
                                           device=device)
    if args.verbose_level >= 0:
        if len(loss_history) ==0:
            print("No checkpoint found, starting fresh training.")
        else:
            print(f"Epochs already trained: {len(loss_history)}")

    _ = ut.get_header(params, paths, verbose=True and args.verbose_level > 1)
    
    # ── Training ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("[dry_run] Dry run enabled — skipping training loop.")
        return loss_history, ""
    loss_history, log_messages = train(
        train_loader            = train_loader,
        val_loader            = val_loader,
        anomaly_loader        = anomaly_loader,
        Enc=Enc, Dec=Dec, Dis=Dis,
        optEncDec=optEncDec, optDis=optDis,
        reconstruction_loss_fn  = reconstruction_loss_fn,
        adversarial_loss_fn     = adversarial_loss_fn,
        loss_history            = loss_history,
        args                    = args,
        params=params, paths=paths, suffix=suffix,
        device=device,
        verbose_print           = True and args.verbose_level > 0,
        verbose_level           = args.verbose_level,
        model_save_interval     = args.model_save_interval,
        data_train_fx           = data_train_fx,
        save_figures            = args.save_figures,
    )

    # ── Final save ───────────────────────────────────────────────────────
    # utmc.save_model(Enc, Dec, Dis, optEncDec, optDis, paths, loss_history, suffix, verbose=True and args.verbose_level > 0)
    utmc.save_model(Enc=Enc,Dec=Dec,D=Dis,
                    optEncDec=optEncDec,
                    optD=optDis,
                    loss_history=loss_history,
                    path_models=paths.path_models,
                    suffix=f"{params.subgroup}_{params.latent_dims}",
                    epoch=params.epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    params={"latent_dim": params.latent_dims,
                            "lr_encdec": params.learning_rate_enc_dec,
                            "lr_d": params.learning_rate_dis,
                            "image_size": (params.input_shape[1],params.input_shape[-1]) ,
                            "beta_gan": params.beta_gan,
                            "beta_kl": params.beta_kl,
                            },
                    augmentation=train_loader.dataset.transform,
                    dataset_name= f'{args.dataset_source}_{args.dataset_version}_{args.dataset_cam_type}', #"MVTec_hazelnut",
                    train_dir=paths.train_dir_processed_subgroup,   
                    notes="VAE-GAN trained on normal images only",
                    verbose=True)
    
    ut.save_log_file(f'{paths.path_results_cloud}log_file_{suffix}.txt', log_messages, verbose= args.verbose_level > 0)
    
    print(f"[safety_area] Done: {safety_area}")


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if args.verbose_level > 0:
        print(f"[device] Using: {device}")
    torch.autograd.set_detect_anomaly(True)

    # ── Resolve safety areas to train ────────────────────────────────────
    if args.safety_area.upper() == "ALL":
        areas_to_train = ALL_SAFETY_AREAS
        if args.verbose_level > 0:
            print(f"[safety_area] ALL selected → training {len(areas_to_train)} areas: {areas_to_train}")
    else:
        areas_to_train = [args.safety_area]
    if args.verbose_level > 0:
        print('-'*80)
    ut.get_time(suff="start")

    for idx, area in enumerate(areas_to_train):
        if _STOP_TRAINING:
            print(f"[INFO] Stop flag set — skipping remaining areas: {areas_to_train[idx:]}")
            break
        if args.verbose_level > 0:
            print(f"\n[progress] Area {idx + 1}/{len(areas_to_train)}: {area}")
        train_one_safety_area(area, args, device)

    print("\nAll training complete.")
    ut.get_time(suff="end")


if __name__ == "__main__":
    main()
