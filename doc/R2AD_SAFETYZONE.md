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

No resizing takes place during preprocessing. The default `--transfer manifest` writes CSV references to original frames, without copying images or creating symlinks. The original files must remain available at their recorded absolute paths. `--transfer symlink` creates linked image trees; `--transfer copy` physically copies images.

Manifest output under `V6/r2ad/full_frames/`: `train.csv`, `val.csv`, `test.csv`, `normal_test.csv`, combined `manifest.csv`, and `metadata.json`. Each row records its split, label, and original source path. Unexpected frames occur only in `test.csv`; `normal_test.csv` contains the normal held-out test frames. Existing outputs must be empty to prevent accidentally combining experiment runs; use a new `--output-dir` for a new split.

Set the existing `r2ad` YAML keys to the generated lists (the script prints their full paths):

```yaml
r2ad:
  train_frames: r2ad/full_frames/train.csv
  val_frames: r2ad/full_frames/val.csv
  test_frames: r2ad/full_frames/test.csv
  normal_test_frames: r2ad/full_frames/normal_test.csv
```

These relative paths resolve under `data.dataset_base`. Training, calibration, evaluation, and stress testing accept these CSV files through the same settings and CLI path arguments as image directories. Keep `--dataset-source images` (the default); the MCAP manifest is a separate format. CSV source paths can also be relative to the CSV file. Use the individual split files for training/evaluation, rather than the combined manifest.

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
