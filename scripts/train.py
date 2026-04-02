"""
train_CAD.py
------------
VAE-GAN training script for Collaborative Anomaly Detection (CAD).
Converted from notebook: N4_train_CAD_updated_model_train_s1.ipynb

Architecture : Encoder + Decoder (VAE) + Discriminator (GAN)
Loss         : Reconstruction (MSE) + KL divergence + Adversarial (BCEWithLogits)
"""

import os
import json
import math
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

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
    verbose: bool = True,
):
    """Build a deterministic train/val split and save it as JSON."""
    base_dataset = datasets.ImageFolder(root=root_dir)
    samples      = list(base_dataset.samples)

    samples_sorted = sorted(enumerate(samples), key=lambda x: x[1][0])

    train_indices, val_indices = [], []
    for sorted_idx, (old_idx, _) in enumerate(samples_sorted):
        if sorted_idx % val_every == val_offset:
            val_indices.append(old_idx)
        else:
            train_indices.append(old_idx)

    split_info = {
        "root_dir":     str(root_dir),
        "rule":         {
            "type":        "modulo",
            "val_every":   int(val_every),
            "val_offset":  int(val_offset),
            "description": f"sorted_index % {val_every} == {val_offset} -> val, else train",
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    train_loader,
    val_loader,
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

    iterator = tqdm(
        range(params.epochs),
        initial     = len(loss_history),
        total       = params.epochs,
        desc        = "Training Epochs",
        position    = 0,
        leave       = False,
    )

    for iter_num_id, iter_num in enumerate(iterator):

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

        use_dis_for_vae_training = True
        train_dis                = True

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

            annealing_lambda = 1.0  # set to min(1.0, epoch/total) to enable KL annealing

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
                lossDis_real  = adversarial_loss_fn(real_logits, torch.ones_like(real_logits))

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

        epoch_loss["dis_acc"] = accuracy_score(dis_labels, dis_preds)
        epoch_loss["dis_F1"]  = f1_score(dis_labels, dis_preds)
        loss_history.append(epoch_loss)

        # ── Visualisation / monitoring ───────────────────────────────────────
        if save_figures:
            colormap_anomaly_map = ut.get_colormap()  # noqa: F841
            ut.plot_images(
                real_images, reconstructed_images, len(loss_history), paths,
                plot_anomaly_scores=True, save_fig=True, interval=5, destroy_fig=True,
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
        if iter_num_id == 1:
            end_time_first = datetime.now()
            eta = (end_time_first - start_time_first) * pending_epochs
            if args.verbose_level > 1:
                print(
                f"[timing] 1 epoch = {end_time_first - start_time_first} "
                f"→ {pending_epochs} remaining epochs ≈ {eta}"
                )

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
ALL_SAFETY_AREAS = ["RoboArm", "ConvBelt", "PLeft", "PRight"]


def parse_args():
    p = argparse.ArgumentParser(description="Train VAE-GAN for CAD anomaly detection")
    p.add_argument("--safety_area",         default="RoboArm",
                   help="Safety area (subgroup) to train. Pass 'ALL' to train every area sequentially.")
    
    p.add_argument("--dataset_source",     default="SR",       help="Dataset version tag")
    p.add_argument("--dataset_version",     default="v2",       help="Dataset version tag")
    p.add_argument("--dataset_cam_type",        default="refined", help="Camera / dataset type")
    p.add_argument("--mask_image_name",     default=3015,   type=int)
    p.add_argument("--epochs",              default=200,   type=int)
    p.add_argument("--batch_size",          default=16,     type=int)
    p.add_argument("--latent_dims",         default=64,     type=int)
    p.add_argument("--exp_type",            default="E3", help="Experiment type key for ut.get_parameters_by_experiment")
    p.add_argument("--augmentation_type",   default="custom", choices=["min", "custom"])
    p.add_argument("--num_workers",         default=0,      type=int)
    p.add_argument("--pin_memory",          default=False,  type=bool)
    p.add_argument("--val_every",           default=5,      type=int, help="1-in-N frames goes to validation")
    p.add_argument("--val_offset",          default=4,      type=int)
    p.add_argument("--force_rebuild_split", action="store_true", help="Force rebuild the train/val split JSON")
    p.add_argument("--model_override",      action="store_true", help="Rename existing checkpoint before training")
    p.add_argument("--model_save_interval", default=10,     type=int)
    p.add_argument("--verbose_level",       default=0,      type=int, choices=[0, 1, 2])
    p.add_argument("--save_path_type",      default="cloud", choices=["cloud", "local"])
    p.add_argument("--checkpoints",
                   default="scripts/dm_checkpoints/checkpoints_32", 
                   choices=["scripts/dm_checkpoints/checkpoints_32", 
                            "scripts/dm_checkpoints/checkpoints_33"])
    p.add_argument("--save_figures",        action="store_true", default=False,
                   help="Save reconstruction & tracking figures during training. "
                        "When disabled only loss curves (results/training) and model "
                        "checkpoints (results/models) are written.")
    return p.parse_args()


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

    params = ut.get_parameters_by_experiment(params, verbose=True and args.verbose_level > 0)
    _ = ut.get_header(params, paths, verbose=True and args.verbose_level > 1)

    # ── Output dirs ───────────────────────────────────────────────────────
    paths.path_codes_cloud   = paths.path_codes
    paths.path_codes_main    = os.path.join(paths.path_codes, 'scripts')
    paths.path_codes_local   = os.path.join(paths.path_results_local, 'scripts')
    paths.path_results_cloud = os.path.join(paths.path_codes_cloud, 'scripts/results')
    paths.history_fname      = "vae_gan_train_history.csv"
    os.makedirs(paths.path_codes_main, exist_ok=True)

    # Fixed output dirs used regardless of save_figures flag
    paths.path_training_curves = os.path.join(paths.path_codes_main, "results", "training")
    # paths.path_models      = os.path.join(paths.path_codes_main, args.checkpoints)
    paths.path_models      = os.path.join(os.getcwd(), args.checkpoints)
    os.makedirs(paths.path_training_curves, exist_ok=True)
    os.makedirs(paths.path_models,          exist_ok=True)
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
        dir            = 'scripts/results',
        verbose        = True  and args.verbose_level > 1,
    )
    paths.suffix = suffix
    # ── Data split ────────────────────────────────────────────────────────
    split_save_path = os.path.join(
        paths.train_dir_processed_subgroup,
        f"split_4train_1val_{safety_area}.json",
    )
    prepare_or_load_video_split(
        train_dir_processed_subgroup = paths.train_dir_processed_subgroup,
        split_save_path  = split_save_path,
        val_every        = args.val_every,
        val_offset       = args.val_offset,
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

    paths.train_classes     = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
    paths.class_names_train = train_dataset.classes
    if args.verbose_level >= 0:
        print(f"Samples (Train/Validation) : {len(train_dataset)} / {len(val_dataset)}")
        print(f"Data Loaded from : {paths.train_dir_processed_subgroup}")
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
        print(f"Epochs already trained: {len(loss_history)}")

    _ = ut.get_header(params, paths, verbose=True and args.verbose_level > 1)
    
    # ── Training ─────────────────────────────────────────────────────────
    loss_history, log_messages = train(
        train_loader            = train_loader,
        val_loader            = val_loader,
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
    
    ut.save_log_file(f'paths.log_file_full_{suffix}', log_messages, verbose= args.verbose_level > 0)
    
    print(f"[safety_area] Done: {safety_area}")


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
