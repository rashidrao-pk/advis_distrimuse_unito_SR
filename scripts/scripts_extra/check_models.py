import os
import json
import argparse
from types import SimpleNamespace

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

import utils as ut
import utils_model as utmc
from utils_model import Encoder, Decoder, Discriminator


ALL_SAFETY_AREAS = ["RoboArm", "ConvBelt", "PLeft", "PRight"]


def tensor_to_hwc_float32(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    arr = arr * 0.5 + 0.5
    return np.clip(arr, 0.0, 1.0)


def save_rgb_float_image(path: str, img_float01: np.ndarray) -> None:
    img_uint8 = (np.clip(img_float01, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_threshold(threshold_dir: str, safety_area: str) -> float:
    json_path = os.path.join(threshold_dir, safety_area, f"threshold_{safety_area}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return float(d["threshold"])
    raise FileNotFoundError(f"Threshold file not found: {json_path}")


def build_suffix_for_area(area, args):
    params = SimpleNamespace()
    params.subgroup = area
    params.latent_dims = args.latent_dims
    params.z_dim = args.latent_dims
    params.dataset_type = args.dataset_source_name
    params.subgroup_mask = args.subgroup_mask
    params.target_size = (128, 128)
    params.epochs = getattr(args, "epochs", 0)

    if not hasattr(args, "save_figures"):
        args.save_figures = False
    if not hasattr(args, "train"):
        args.train = False
    if not hasattr(args, "test"):
        args.test = False
    if not hasattr(args, "inference"):
        args.inference = True

    paths = SimpleNamespace()
    cwd = os.getcwd()
    paths.path_codes = cwd
    paths.path_codes_local = cwd
    paths.path_results_local = cwd
    paths.path_results_cloud = cwd
    paths.path_models = os.path.join(cwd, args.checkpoints)

    suffix, _ = ut.get_create_results_path(
        area,
        params,
        args,
        paths,
        save_path_type=args.save_path_type,
        dir="scripts/results",
        create_dirs = True,
        verbose=False,
    )
    return suffix


def make_plot_paths(area: str, args, demo_dir: str):
    params = SimpleNamespace()
    params.subgroup = area
    params.epochs = 0
    params.learning_rate_enc_dec = getattr(args, "learning_rate_enc_dec", 0.0)
    params.learning_rate_dis = getattr(args, "learning_rate_dis", 0.0)

    paths = SimpleNamespace()
    paths.dataset_version = getattr(args, "dataset_version", "unknown")
    paths.dataset_type = getattr(args, "dataset_source_name", "unknown")
    paths.path_results = demo_dir
    paths.suffix = f"check_{area}"

    return params, paths


def plot_loss_sep(
    loss_history,
    params,
    paths,
    plot_type=3,
    save_fig=True,
    destroy_fig=True,
    verbose_print=False,
    plot_long_header=True,
    fontsize=12,
):
    if not loss_history:
        print(f"[WARN] Empty loss_history for suffix={paths.suffix}, skipping plot.")
        return

    n_epochs = len(loss_history)
    epochs = np.arange(1, n_epochs + 1)

    def get_series(key, default=np.nan):
        vals = []
        for l in loss_history:
            vals.append(l.get(key, default) if isinstance(l, dict) else default)
        return vals

    if plot_type == 3:
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    elif plot_type == 2:
        fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    else:
        raise ValueError("plot_type must be 2 or 3")

    ax = axs[0]
    ax.plot(epochs, get_series("recon_loss"), label="Reconstruction Loss")
    ax.plot(epochs, get_series("kl_loss"), label="KL Loss")
    ax.plot(epochs, get_series("beta_kl_loss"), label="beta * KL Loss")
    ax.plot(epochs, get_series("gan_loss"), label="GAN Loss")
    ax.plot(epochs, get_series("beta_gan_loss"), label="beta * GAN Loss")
    ax.plot(epochs, get_series("vae_loss"), label="VAE Loss")
    ax.plot(epochs, get_series("annealing_lambda"), label="Annealing")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VAE Loss (Log)")
    ax.set_yscale("log")
    ax.set_xlim(1, n_epochs)
    ax.legend()

    ax = axs[1]
    ax.scatter(
        epochs,
        get_series("dis_acc"),
        label="Accuracy(Dis)",
        alpha=0.5,
        marker="+",
    )
    ax.scatter(
        epochs,
        get_series("dis_F1"),
        label="F1(Dis)",
        alpha=0.5,
        marker="x",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator Scores")
    ax.set_xlim(1, n_epochs)
    ax.legend()

    if plot_type == 3:
        ax = axs[2]
        ax.plot(epochs, get_series("gan_loss"), label="GAN Loss")
        ax.scatter(
            epochs,
            get_series("disc_loss"),
            label="Discriminator Loss",
            s=8,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("GAN/Disc Loss")
        ax.set_xlim(1, n_epochs)
        ax.legend()

    tick_step = max(1, n_epochs // 10)
    x_ticks = list(range(1, n_epochs + 1, tick_step))
    if x_ticks[-1] != n_epochs:
        x_ticks.append(n_epochs)
    for ax in axs:
        ax.set_xticks(x_ticks)

    header_top = f"Loss Progression (Loaded Epochs: {n_epochs})"
    if plot_long_header:
        column_headers = ["DS_v", "camera", "subgroup", "Epochs"]
        column_widths = [12, 12, 12, 14]
        total_epochs = getattr(params, "epochs", n_epochs)
        values = [
            str(paths.dataset_version),
            str(paths.dataset_type),
            str(params.subgroup),
            f"{n_epochs}/{total_epochs}",
        ]

        header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(column_headers, column_widths)) + " |"
        value_row = "| " + " | ".join(str(v).ljust(w) for v, w in zip(values, column_widths)) + " |"
        ttl = f"{header_top}\n{header_row}\n{value_row}"
    else:
        ttl = header_top

    plt.suptitle(ttl, fontsize=fontsize)
    plt.tight_layout()

    if save_fig:
        out_path = os.path.join(paths.path_results, f"history_{paths.suffix}.png")
        plt.savefig(out_path, bbox_inches="tight")
        print(f"[OK] saved learning curve: {out_path}")

    if destroy_fig:
        plt.close(fig)
    else:
        plt.show()


def create_dummy_input(device, mode="random"):
    if mode == "zeros":
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    elif mode == "ones":
        dummy = np.ones((128, 128, 3), dtype=np.uint8) * 255
    elif mode == "gradient":
        x = np.linspace(0, 255, 128, dtype=np.uint8)
        y = np.linspace(0, 255, 128, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        dummy = np.stack([xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)], axis=-1)
    else:
        dummy = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    tensor = torch.from_numpy(dummy).permute(2, 0, 1).float() / 255.0
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tensor = normalize(tensor).unsqueeze(0).to(device)
    return dummy, tensor


def load_model_bundle(area, args, device):
    enc = Encoder(z_size=args.latent_dims).to(device)
    dec = Decoder(z_size=args.latent_dims).to(device)
    dis = Discriminator().to(device)

    optED, optD = utmc.get_optimizers(enc, dec, dis, verbose=False)
    suffix = build_suffix_for_area(area, args)
    checkpoint_root = os.path.join(os.getcwd(), args.checkpoints)

    print("=" * 100)
    print(f"[LOAD] area={area}")
    print(f"[LOAD] checkpoint_root={checkpoint_root}")
    print(f"[LOAD] suffix={suffix}")

    history,config = utmc.load_model(
        enc, dec, dis, optED, optD,
        checkpoint_root, suffix, device=device, verbose=True
    )

    if history is None or len(history) == 0:
        raise RuntimeError(f"No checkpoint found or empty history for area={area}, suffix={suffix}")

    enc.eval()
    dec.eval()
    dis.eval()

    tau = None
    try:
        tau = load_threshold(args.threshold_dir, area)
        print(f"[OK] threshold for {area}: {tau:.6f}")
    except Exception as e:
        print(f"[WARN] threshold could not be loaded for {area}: {e}")

    return {
        "encoder": enc,
        "decoder": dec,
        "discriminator": dis,
        "history": history,
        "suffix": suffix,
        "threshold": tau,
    }


def run_dummy_reconstruction(area, bundle, dummy_tensor, dummy_rgb_uint8, demo_dir):
    enc = bundle["encoder"]
    dec = bundle["decoder"]

    print(f"[TEST] Running dummy inference for area={area}")

    with torch.no_grad():
        mu, logvar = enc(dummy_tensor)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = dec(z)

    print(f"[SHAPE] input   : {tuple(dummy_tensor.shape)}")
    print(f"[SHAPE] mu      : {tuple(mu.shape)}")
    print(f"[SHAPE] logvar  : {tuple(logvar.shape)}")
    print(f"[SHAPE] z       : {tuple(z.shape)}")
    print(f"[SHAPE] recon   : {tuple(recon.shape)}")

    recon_hwc = tensor_to_hwc_float32(recon.squeeze(0))
    inp_hwc = tensor_to_hwc_float32(dummy_tensor.squeeze(0))

    mse = float(np.mean((inp_hwc - recon_hwc) ** 2))
    mae = float(np.mean(np.abs(inp_hwc - recon_hwc)))

    print(f"[OK] reconstruction successful for {area}")
    print(f"[METRIC] MSE={mse:.8f} | MAE={mae:.8f}")

    area_dir = os.path.join(demo_dir, area)
    ensure_dir(area_dir)

    orig_path = os.path.join(area_dir, "dummy_input.png")
    recon_path = os.path.join(area_dir, "dummy_reconstruction.png")
    comp_path = os.path.join(area_dir, "comparison.png")

    save_rgb_float_image(orig_path, inp_hwc)
    save_rgb_float_image(recon_path, recon_hwc)

    comparison = np.hstack([
        cv2.cvtColor(dummy_rgb_uint8, cv2.COLOR_RGB2BGR),
        cv2.cvtColor((recon_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    ])
    cv2.imwrite(comp_path, comparison)

    print(f"[OK] saved original       : {orig_path}")
    print(f"[OK] saved reconstruction : {recon_path}")
    print(f"[OK] saved comparison     : {comp_path}")


def parse_args():
    p = argparse.ArgumentParser("Check trained models and plot learning curves")

    p.add_argument("--safety_area", default="ALL", help="ALL or one of RoboArm, ConvBelt, PLeft, PRight")
    p.add_argument("--threshold_dir", required=True)
    p.add_argument("--checkpoints", default="scripts/results/models")
    p.add_argument("--latent_dims", type=int, default=64)

    p.add_argument("--dataset_source_name", default="SR")
    p.add_argument("--subgroup_mask", default="MASK")
    p.add_argument("--save_path_type", default="local")
    p.add_argument("--save_figures", action="store_true", default=False)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--dataset_version", default="unknown")
    p.add_argument("--learning_rate_enc_dec", type=float, default=0.0)
    p.add_argument("--learning_rate_dis", type=float, default=0.0)

    p.add_argument("--dummy_mode", default="random", choices=["random", "zeros", "ones", "gradient"])
    p.add_argument("--demo_dir", default="scripts/results/demo")

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.demo_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device={device}")

    areas = ALL_SAFETY_AREAS if args.safety_area.upper() == "ALL" else [args.safety_area]

    dummy_rgb, dummy_tensor = create_dummy_input(device=device, mode=args.dummy_mode)

    summary = []
    for area in areas:
        try:
            bundle = load_model_bundle(area, args, device)
            
            run_dummy_reconstruction(area, bundle, dummy_tensor, dummy_rgb, args.demo_dir)

            plot_params, plot_paths = make_plot_paths(area, args, args.demo_dir)
            plot_params.epochs = len(bundle["history"])
            print('plot_paths-->', plot_paths)  # Debug: print the paths used for plotting
            plot_loss_sep(
                bundle["history"],
                plot_params,
                plot_paths,
                plot_type=3,
                save_fig=True,
                destroy_fig=True,
                plot_long_header=True,
            )

            summary.append((area, "OK", len(bundle["history"])))
        except Exception as e:
            print(f"[ERROR] area={area} failed: {e}")
            summary.append((area, f"FAILED: {e}", 0))

    print("\n" + "=" * 100)
    print("SUMMARY")
    for area, status, n_epochs in summary:
        print(f" - {area:10s} | {status} | loaded_epochs={n_epochs}")
    print("=" * 100)


if __name__ == "__main__":
    main()