# R²AD – direct MCAP data, 4:1 within each scenario

This is the **MCAP-native extension** for `advis_distrimuse_unito_SR`.
It keeps the current ADVIS production scripts unchanged. It does not extract,
copy, or resize any camera frames on disk.

## Install

In your existing `pt_312` environment:

```bash
python -m pip install mcap mcap-ros2-support
python -m pytest tests/test_mcap_manifest.py -q
```

Run commands from the repository root after copying the package contents there.
The MCAP code decodes ROS2 `sensor_msgs/msg/CompressedImage` (JPEG/PNG) and
standard 8-bit `Image` encodings with OpenCV. No `cv_bridge` or Pixi required.

## 1. One-time indexing (no image extraction)

```bash
python scripts/index_rosbags_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --all --camera back_view
```

Or a small smoke-test:

```bash
python scripts/index_rosbags_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --scenario 1_0 3_0 8_0 --camera back_view --max-frames 30
```

`--all` uses YAML `scenario_options` and `scenario_categories`.
The default output is `<data.dataset_base>/r2ad/manifest.csv`. You may set
`r2ad.manifest` in the existing YAML, or pass `--output`.

Manifest fields include `scenario`, `label`, `split`, `bag`, `topic`,
`log_time` (nanoseconds), `occurrence` (tie-breaker for same-timestamp
messages), and `ordinal`. No frame bytes are written.

**4:1 split:** In **each normal scenario**, the selected sample indices
`0,1,2,3 → train`; `4 → val`, repeating. Scenarios 1_0 and 3_0
both contribute train and validation frames. Scenarios classified as unexpected
are strictly reserved for `test`.

`--save-every-n` subsamples the indexed stream. `--max-frames` caps each
scenario's indexed frames (useful for initial tests). The CSV is overwritten
on each indexing run, so run `--all` before full training.

**Important statistical note:** adjacent video frames are highly correlated.
The 4:1 split is good for checking reconstruction training and retaining action
coverage, but does NOT measure generalization to unseen actions or independent
camera sessions. For publication, evaluate on separately recorded normal
sequences and disjoint unexpected scenarios without training contamination.

## 2. Train using MCAP directly

```bash
python scripts/train_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --dataset-source mcap \
  --safety-area SafetyZone \
  --patch-size 224 --samples-per-frame 6 \
  --epochs 200 --batch-size 32 --workers 4
```

Uses existing `data.dataset_base`, `r2ad.safety_zone_mask`,
`models.latent_dims`, `training.*`, and `results.output_dir`.
The SafetyZone mask must be at the source frame resolution.
Each frame is decoded on demand, sampled into an arbitrary random rectangle,
then resized to the existing 128×128 VAE-GAN input. The GPU never sees full
resolution images; PyTorch DataLoader workers do the crop/resize on CPU.

**I/O performance:** Indexed reads are efficient when MCAP contains seekable
chunk indexes. Random access can be slow for unindexed files and with heavy
zstd decompression. The reader keeps at most two open bag file handles per
DataLoader worker; tune `--workers`, `--save-every-n`, and storage locality.
A follow-on optimization is a bounded decoded-frame LRU cache or sequential
chunk-level sampling.

## 3. Calibrate from all normal scenario validation frames

```bash
python scripts/evaluate_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --dataset-source mcap --safety-area SafetyZone \
  --calibrate --patch-size 224 --stride 64
```

## 4. Test on unexpected scenarios

```bash
python scripts/evaluate_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --dataset-source mcap --safety-area SafetyZone \
  --split test --patch-size 224 --stride 64 --save-maps --max-frames 50
```

Optional YAML addition:

```yaml
r2ad:
  dataset_version: V6
  manifest: r2ad/manifest.csv
  safety_zone_mask: /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_SafetyZone.png
```

The original `data.training` and `data.testing` paths continue serving the
old 128×128 crop pipeline. Existing calibration thresholds cannot be reused
for R²AD's stitched residual anomaly score.

## Constraints and limitations

- Only `sensor_msgs/msg/CompressedImage` and common 8-bit `Image` encodings are supported.
- MCAP bag files must exist at the paths declared in the YAML.
- More than one image message with the exact same topic/log timestamp is
  handled via the `occurrence` index.
- The first experiment **does not** include camera registration.
- Runtime wasn't tested against a real MCAP file in this package-building
  environment; test on one short scenario before a full run.
- The original image-folder mode remains available using `--dataset-source images`.
