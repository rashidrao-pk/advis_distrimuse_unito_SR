# R²AD — Randomized Rectangle Anomaly Detection (experimental V1)

For current annotation-aware virtual splits and the complete V6 SafetyZone training workflow, use the [training runbook and status checklist](R2AD/TRAINING_RUNBOOK.md). The examples below include historical directory-based setup instructions.

**Status:** self-contained research prototype, designed to be copied into
`rashidrao-pk/advis_distrimuse_unito_SR` on the
`feature/r2ad-random-rectangles` branch. **Does not edit any existing source file.**

## What this tests

- Training on **normal full-resolution frames** using online randomized rectangles from ONE binary safety-area mask.
- Existing-compatible **128×128 VAE-GAN encoder/decoder architecture**, latent 64 by default. New checkpoints are separate.
- Inference on **fixed overlapping source rectangles** (default 224×224, stride 64), batched model evaluation, Hann-weighted stitching.
- Independent score calibration using stitched RGB mean absolute reconstruction residual + spatial q99. **This is NOT the production TAAS distance calculation.** Do not mix its thresholds with original ADVIS thresholds.
- Synthetic camera shift robustness study and optional labeled classification report.

## Crucial input requirements

Use **uncropped original frames** (same dimensions as your mask, ideally 1280×720) — **NOT** existing 128×128 per-area cropped training PNGs. Mask must have the same `(height,width)` as the incoming original frame. Set paths to the corresponding V6 full-frame images you generated from ROS bags. Train/validation/test must be **distinct recording groups/scenarios**, not neighboring frames from the same capture: random frame splits inflate scores. Only **normal frames** should enter training and threshold calibration.

Example layout (illustrative only):

```
/mnt/v6/full_frames/
  train/normal/scenario_01/frame_*.png
  train/normal/scenario_02/frame_*.png
  val/normal/scenario_03/frame_*.png
  test/normal/scenario_04/frame_*.png
  test/anomaly/scenario_13/frame_*.png
```

## Install

From repository root:

```bash
# copy r2ad/, scripts/*r2ad.py, scripts/compare_scores.py into your repository
# use existing Pixi or conda PyTorch environment
python -m pip install -r requirements-extra.txt
python -m pytest tests/test_core.py -q
```



## Train

```bash
python scripts/train_r2ad.py \
  --train-dir /path/to/full_frames/train/normal \
  --val-dir /path/to/full_frames/val/normal \
  --mask /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_PLeft.png \
  --output results/V6/r2ad/PLeft/model_r2ad.pt \
  --patch-size 224 --samples-per-frame 6 --min-coverage 0.85 \
  --epochs 50 --batch-size 32 --workers 4
```

## Calibrate on NORMAL held-out validation recordings

```bash
python scripts/evaluate_r2ad.py \
  --input /path/to/full_frames/val/normal \
  --mask /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_PLeft.png \
  --checkpoint results/V6/r2ad/PLeft/model_r2ad.pt \
  --output-dir results/V6/r2ad/PLeft/calibration \
  --calibrate --threshold-percentile 99 \
  --patch-size 224 --stride 64 --min-coverage 0.85
```

## Inference / anomaly heatmaps

```bash
python scripts/evaluate_r2ad.py \
  --input /path/to/full_frames/test \
  --mask /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_PLeft.png \
  --checkpoint results/V6/r2ad/PLeft/model_r2ad.pt \
  --threshold-json results/V6/r2ad/PLeft/calibration/threshold.json \
  --output-dir results/V6/r2ad/PLeft/test \
  --patch-size 224 --stride 64 --min-coverage 0.85 --save-maps
```

The output contains `scores.csv`, `*_overlay.png`, `*_heat.png`, and stitched anomaly-map `*.npy` files. To avoid expensive I/O, omit `--save-maps` for large evaluations.

## Camera displacement stress test (NORMAL frames only)

```bash
python scripts/stress_test_r2ad.py \
  --input /path/to/full_frames/test/normal \
  --mask /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_PLeft.png \
  --checkpoint results/V6/r2ad/PLeft/model_r2ad.pt \
  --threshold-json results/V6/r2ad/PLeft/calibration/threshold.json \
  --output-dir results/V6/r2ad/PLeft/robustness --max-frames 30
```

Output `robustness.csv` and FPR plots for x-translation, rotation and scaling. This is **camera-shift-like image perturbation**, not a physically moved camera. Because the mask stays in fixed image coordinates, large shifts can move actual safety areas outside the mask. Treat this as a robustness stress test, not proof of camera alignment.

## Labeled binary metrics

Create `labels.csv` with **exact absolute/full path strings** matching `scores.csv`:

```csv
file,label
/path/to/test/normal/frame1.png,0
/path/to/test/anomaly/frame2.png,1
```

```bash
python scripts/compare_scores.py \
 --scores results/V6/r2ad/PLeft/test/scores.csv \
 --labels /path/to/labels.csv \
 --threshold-json results/V6/r2ad/PLeft/calibration/threshold.json \
 --out results/V6/r2ad/PLeft/test/metrics.json
```

## Research controls and limitations

1. **Control:** Compare to existing ADVIS on identical **held-out source frames**, same ROI and ground truth, and same calibration split, but separately recalibrated thresholds.
2. **Ablations:** `--patch-size` 128/192/224; `--stride` 32/64/96; coverage 0.5/0.7/0.85. Source patch size must not exceed frame size; thinner masks may not support 0.85 coverage.
3. **Coverage:** Report `covered_mask_fraction` because center-coverage filtering leaves some boundaries uncovered. Never claim detection outside the covered region.
4. **No global context:** Patch reconstruction can miss spatial relationships; test contextual anomalies explicitly.
5. **Model adaptation:** Model input is always 128×128. The 224 is the source rectangle, not network resolution. Re-use architecture, **not original area-specific weights**, because data distribution has changed. You may later fine-tune separately, but baseline from scratch is safer.
6. **Determinism:** The fixed-grid inference and calibration are deterministic; online training is stochastic, with a seed for process-level reproducibility. For strict cross-worker reproducibility add deterministic worker seeding and CPU/CUDA determinism settings.
7. **GPU memory:** VAE-GAN's original fully connected layers are memory heavy. Reduce `--batch-size` to 8 or 16 if OOM. CPU evaluation may be slow.
8. **Scope:** This experiment is independent of ROS2/Zenoh, camera-registration, and existing production TAAS code. Integrate the winner only after evaluating it offline.
9. **Safety:** This is a research AD prototype and must not substitute for certified industrial robot safety systems.

## Suggested repository placement

```
r2ad/core.py
r2ad/dataset.py
scripts/train_r2ad.py
scripts/evaluate_r2ad.py
scripts/stress_test_r2ad.py
scripts/compare_scores.py
tests/test_core.py
```

## R²AD V1: Using the existing ADVIS configuration

The V1 scripts now read `configs/cf_dataset_epito.yaml` with `--config` (default)
plus `--safety-area PLeft|PRight|RoboArm|ConvBelt`.

Existing configuration keys used **without duplication**:
- `data.dataset_base`: root for relative R2AD image directories.
- `data.mask_types.<area>`: the actual full-frame binary mask.
- `models.latent_dims`: model latent dimensions.
- `training.epochs`, `training.batch_size`, `training.num_workers`: training defaults.
- `results.output_dir`: base output directory.

Add this **small block** to the *end* of your existing `configs/cf_dataset_epito.yaml`, replacing only the `REPLACE_ME` paths. These must point to actual full-resolution original frames, not 128×128 crops. Paths may be absolute or relative to `data.dataset_base`:

```yaml
r2ad:
  dataset_version: V6
  train_frames: REPLACE_ME/full_frames/train/normal
  val_frames: REPLACE_ME/full_frames/val/normal
  test_frames: REPLACE_ME/full_frames/test
  normal_test_frames: REPLACE_ME/full_frames/test/normal
```

**Do not** replace `data.training` or `data.testing`: the production ADVIS code
still needs those existing cropped datasets. For meaningful metrics split recordings
by scenario (not adjacent frames) and train/calibrate only on normal recordings.

Run in repository root:

```bash
python scripts/train_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area PLeft \
  --patch-size 224 --samples-per-frame 6 --epochs 50 --batch-size 32

python scripts/evaluate_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area PLeft \
  --calibrate --patch-size 224 --stride 64

python scripts/evaluate_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area PLeft \
  --patch-size 224 --stride 64 --save-maps

python scripts/stress_test_r2ad.py --config configs/cf_dataset_epito.yaml \
  --safety-area PLeft --max-frames 30
```

Defaults resolve to these destinations under `results.output_dir`:
`<result_base>/V6/r2ad/<area>/model_r2ad.pt`,
`.../calibration/threshold.json`, `.../test/`, and `.../robustness/`.
You can override all resolved paths with `--train-dir`, `--val-dir`, `--input`,
`--mask`, `--checkpoint`, `--output`, `--output-dir`, or `--threshold-json`.

**The earlier commands using explicit paths still work** when `--safety-area`
is supplied. The R²AD output score is the original suite's stitched reconstruction
residual, *not* existing TAAS, and needs its own validation threshold.
