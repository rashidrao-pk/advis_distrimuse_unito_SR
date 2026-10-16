# R²AD V1: Unified SafetyZone preprocessing

Uses the existing `data.mask_types` keys `PLeft`, `PRight`, `RoboArm`, and `ConvBelt`, merging positive pixels with logical OR. Masks MUST share identical frame resolution; no silent rescaling. The output SafetyZone mask is saved under the V6 `masks/` folder. A preview shows component outlines.

## 1. Create the mask (no frames required)

```bash
python scripts/preprocess_r2ad.py --config configs/cf_dataset_epito.yaml --mask-only
```

## 2. Point the existing config at original camera images

Add the `r2ad:` section from `configs/R2AD_CONFIG_ADD_TO_EXISTING.yaml` to `configs/cf_dataset_epito.yaml`. Edit `r2ad.normal_source` to the true full-frame **normal** scenario root. Optional `unexpected_source` contains anomalous test images (kept out of training/validation).

Expected normal source layout for reproducible **grouped** splits:

```
normal_frames/
  scenario_1/frame_00001.png
  scenario_2/frame_00001.png
  scenario_3/frame_00001.png
  ...
```

There must be at least three independent first-level scenario folders. Entire scenario folders are assigned to training, validation, or test (avoids obvious leakage).

## 3. Prepare full-resolution frame splits

```bash
python scripts/preprocess_r2ad.py --config configs/cf_dataset_epito.yaml
```

No 128x128 resizing takes place during preprocessing. By default the script creates **symlinks**, so it is efficient on Epito. Move/copy the original data carefully if symlinks will not be portable. To physically copy use `--transfer copy`.

Output under `V6/r2ad/full_frames/`: train/normal, val/normal, test/normal, optionally test/unexpected, manifest.csv, metadata.json. The four original masks and the existing V6 train/test directories remain intact. Existing outputs must be empty to prevent accidentally combining experiment runs.

If there are no separate normal recording folders, use `--split-mode temporal` only for a smoke test. **Do not claim statistically independent test results** from temporally neighboring frames. Prefer physically distinct sequences for a publication.

## 4. Train and evaluate on merged mask

```bash
python scripts/train_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area SafetyZone --epochs 50 --batch-size 32 --patch-size 224 --samples-per-frame 6
python scripts/evaluate_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area SafetyZone --calibrate --patch-size 224 --stride 64
python scripts/evaluate_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area SafetyZone --patch-size 224 --stride 64 --save-maps
python scripts/stress_test_r2ad.py --config configs/cf_dataset_epito.yaml --safety-area SafetyZone --max-frames 30
```

All selected patches must lie mostly inside SafetyZone (default 85% occupancy). Irregular/narrow portions may not receive any grid patches; inspect `covered_mask_fraction` in evaluation CSV before claiming full coverage. Use smaller patch or lower occupancy if needed, and recalibrate thresholds after any changes.

**Important:** The unified SafetyZone creates *one* model for the entire combined region, not four independent zone alarms. It can still output a stitched spatial heatmap; per-area alarms would require a later aggregation step using the original component masks. The original VAE-GAN and TAAS production pipeline are not changed.
