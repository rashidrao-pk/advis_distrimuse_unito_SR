# Smaller patches and measured latent utilization

Updated 2026-11-02. This extends the [training runbook](TRAINING_RUNBOOK.md); it does not overwrite existing run settings or checkpoints.

## First audit the existing model

Latent space is an unbounded Gaussian space, so there is no meaningful percentage of its total volume that is "filled." We report complementary diagnostics on normal held-out patches:

- **Active units:** dimensions with `Var_x(mu_j) > 0.01`. This threshold is configurable and is a heuristic, not a capacity limit.
- **Per-dimension KL:** mean `0.5 * (mu_j² + exp(logvar_j) - 1 - logvar_j)`, in nats. Total KL and mean KL/dimension are both recorded; training uses the latter.
- **Posterior variance and signal/noise:** variation of posterior means compared with mean posterior variance.
- **Effective rank:** participation ratio `(sum eigenvalues)² / sum(eigenvalues²)` and entropy rank of `Cov(mu)`; correlated active dimensions need not represent independent directions.
- **PCA95:** number of components explaining 95% of posterior-mean variance, plus a 2-D projection. The projection is not a map of the entire latent space.

Active-unit diagnostics are used in VAE posterior-collapse analysis; see [He et al., ICLR 2019](https://arxiv.org/abs/1901.05534) and its active-unit evaluation. No single statistic establishes whether larger latent width improves reconstruction or anomaly detection.

On Epito, audit the currently selected best checkpoint before changing width:

```bash
python scripts/audit_r2ad_latents.py \
  --config configs/cf_dataset_epito.yaml \
  --checkpoint results/r2ad/SafetyZone_fast_b32/model_r2ad.pt \
  --input /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --output-dir reports/r2ad_latent_audit_current_validation \
  --samples 2048 --batch-size 64 --device cuda --seed 42 --save-codes
```

Use a specific periodic checkpoint instead if needed. The tool requires normal `val` rows by default, inherits the saved source patch size and validation size-jitter policy, samples without geometric/brightness augmentation, and uses FP32 evaluation. It writes `summary.json`, `dimensions.csv`, `latent_utilization.png`, a README, and optional `latents.npz`. The sample cycles through a seeded random frame order before repeating frames. Inspect the recorded unique-frame/recording counts and increase sample diversity if needed. More samples cannot compensate for using only one recording.

### Actual local check already performed

Only `model_r2ad_epoch_0012.pt` (64-dimensional latent) and one original normal frame were available locally. A deliberately marked exploratory audit used 256 unaugmented random patches from scenario 1_0:

| Diagnostic                          |      Result |
| ----------------------------------- | ----------: |
| Active dimensions at threshold 0.01 |     64 / 64 |
| Participation-ratio effective rank  |       20.14 |
| Entropy effective rank              |       31.40 |
| Components for 95% variation        |          41 |
| Mean KL per dimension               | 5.0546 nats |
| Independent source frames used      |       **1** |

Artifacts: `reports/r2ad_latent_audit_epoch12_exploratory/`. This is not the latest Epito model, not a held-out validation audit, and not evidence that 64 dimensions are saturated. All units are varying on this sample, but some directions are correlated. It does not establish that increasing width will help.

## Smaller source patches with matching inference

### Estimate 32–128 pixel coverage before choosing the next experiment

```bash
python scripts/estimate_r2ad_patch_coverage.py \
  --config configs/cf_dataset_epito.yaml \
  --patch-sizes 32 48 64 96 128 224 \
  --stride-ratios 1 0.5 0.25 --min-coverage 0.85 \
  --output-dir reports/r2ad_patch_coverage
```

Optionally add `--image /path/to/original/frame.png` for overlays on the scene; without it the mask is shown on a gray background. Open `index.html`, `grid_coverage.csv`, `random_coverage.csv`, and `report.json`. Green is covered mask, red is uncovered, and yellow outlines accepted crops. Every source crop still resizes to 128×128 network input. Source patches >=16 are supported; use `--patch-size 64 --fixed-patch-size` or `--patch-size 32 --fixed-patch-size` for genuinely smaller training crops. Validation and inference use that same nominal source size.

The estimator uses the actual inference grid including its end-aligned windows. Counts are accepted grid windows, not a minimum packing solution; even stride=patch can overlap at image edges. `min-coverage` is the fraction of each patch inside SafetyZone, whereas `target-coverage` (default 99%) concerns all SafetyZone pixels. These are different quantities. The optimistic `ceil(mask_pixels/patch_area)` bound ignores boundaries, rejected windows and overlap, so it is not a practical coverage guarantee. Changing grid origin/stride/occupancy can change the result.

Random trials use fixed-size patches on the unaugmented mask, with default budgets 6/24/48/96/192 and three seeds. They estimate union coverage on one frame; they do not guarantee that training's random patches fully cover each frame or predict pixel coverage under changing zoom/translation. Grid CSVs include per-area coverage when component masks are available. Missing component files are explicitly reported; combined-mask estimates still work, or use `--strict-area-masks` to require all four.

The current Mac mask examined for this estimator contains **302,859 positive pixels** in a 1920×1080 frame (14.61%). It differs from the earlier 394,399-pixel mask (19.02%). These counts therefore describe the current local file, not automatically the mask in an older checkpoint or on Epito. The report records the mask checksum; rerun on the target machine and inspect the overlay.

| Source size | Stride | Accepted grid patches | Mask coverage |
|---|---:|---:|---:|
| 32×32 | 16 | 1,101 | 98.53% |
| 32×32 | 8 | 4,289 | 99.20% |
| 48×48 | 24 | 482 | 99.03% |
| 64×64 | 32 | 250 | 96.44% |
| 64×64 | 16 | 973 | 99.32% |
| 96×96 | 24 | 402 | 99.77% |
| 128×128 | 32 | 193 | 97.77% |

On this mask, six random 64×64 patches cover only about **7.63%** on average in the seeded trials; six random 32×32 patches cover about **2.02%**. Smaller crops therefore reduce sampled area per frame at a fixed training sample budget. Increasing samples/frame increases network work, because the model still sees 128×128 inputs. Grid inference can be much more expensive with smaller crops. Upsampling does not create source detail, and very small crops discard context needed for some anomalies.

Smaller source patches can reduce shared scene context, but do not eliminate correlation from neighboring/overlapping patches or adjacent video frames. Preserve recording-group separation and consider temporal subsampling separately; patch size alone cannot establish independent samples. Choose patch size/coverage on normal validation data, then recalibrate the new model.

### Animate smaller crops before training a new model

`scripts/animate_r2ad_patch_sampling.py` needs no checkpoint or split CSV. It visualizes geometry, actual source pixels, the resized 128×128 model input, and the mask inside each crop. It does not produce neural reconstructions or anomaly scores; those require a matching trained checkpoint for a representative evaluation.

```bash
python scripts/animate_r2ad_patch_sampling.py \
  --config configs/cf_dataset_epito.yaml \
  --image /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/sources/normal/scenario_1_0/frame_000000.png \
  --output reports/r2ad_p64_random_zoom_translation.gif \
  --patch-size 64 --sampling random --num-patches 24 \
  --aug-translate 0.005 --aug-rotate 0 --aug-zoom-min 1 --aug-zoom-max 1.03 \
  --aug-contrast 0.05 --aug-brightness 4 --fps 3
```

For cumulative grid placement and coverage, replace the random-sampling options with `--sampling grid --stride 32 --every-n 10` and choose another output filename. All grid patches contribute to coverage; only every tenth is captured to keep playback shorter (plus first/last). Grid mode uses the unaugmented frame; random mode applies the configured camera/brightness transforms and resets coverage per scene. Source crop size is fixed in both modes. Use `--patch-size 32` for smaller crops, with e.g. stride 16 in grid mode.

The local 64-pixel previews already generated are `reports/r2ad_p64_random_zoom_translation.gif` (24 augmented patches) and `reports/r2ad_p64_grid_coverage.gif` (250 grid patches at stride 32, 26 captured GIF frames, 96.44% mask coverage). Each has a final PNG and JSON. Red shading denotes uncovered ROI, not anomaly; no trained model is invoked.

### 128-pixel control command

For the first experiment, use **128×128 source crops**, matching the existing network's 128×128 input. Enable `--fixed-patch-size` so training and validation disable crop-size jitter. Without this flag, the historical 0.9–1.1 crop-size range remains enabled unless manually configured.

```bash
python -u scripts/train_r2ad.py \
  --config configs/cf_dataset_epito.yaml --safety-area SafetyZone \
  --train-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --val-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --output results/r2ad/SafetyZone_p128_z64/model_r2ad.pt \
  --patch-size 128 --fixed-patch-size --latent-dims 64 \
  --epochs 20 --batch-size 32 --workers 8 --device cuda --precision bf16 \
  --aug-translate 0.005 --aug-rotate 0.5 --aug-contrast 0.05 --aug-brightness 4 \
  --samples-per-frame 6 --save-cp-k 1 --latent-audit-every 5
```

```bash
python -u scripts/train_r2ad.py \
  --config configs/cf_dataset_epito.yaml --safety-area SafetyZone \
  --train-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --val-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --output results/r2ad/SafetyZone_p128_z64/model_r2ad.pt \
  --patch-size 128 --fixed-patch-size --latent-dims 64 \
  --epochs 20 --batch-size 32 --workers 8 --device cuda --precision bf16 \
  --aug-translate 0.005 \
  --aug-rotate 0.5 \
  --aug-contrast 0.05 --aug-brightness 4 \
  --samples-per-frame 6 --save-cp-k 1 --latent-audit-every 5
  --aug-translate 0.005 \
  --aug-rotate 0 \
  --aug-zoom-min 1.0 \
  --aug-zoom-max 1.03 \
  --aug-contrast 0.05 \
  --aug-brightness 4 \
  --patch-size 128 \
  --fixed-patch-size \
  --samples-per-frame 6 \
  --save-cp-k 1 \
  --latent-audit-every 5

```

This is a new experiment: do not resume a 224-pixel full-state run while changing its training scale. The network input/output remain 128×128; smaller source crops do not automatically reduce network FLOPs. Smaller patches also lose scene context and cover fewer source pixels per training sample; preserve these facts in comparisons.

Calibration, evaluation, checkpoint inspection and both animation tools now default to the source patch size stored in the checkpoint. Explicit sizes that conflict with recorded training size are rejected. For legacy checkpoints lacking patch metadata the fallback is 224. Default inference stride is 64 for source patches >=224, otherwise one-quarter of patch size (128 → 32); explicit positive strides remain supported. A stride smaller than the patch creates overlapping patches; it is not another patch-size change.

```bash
python scripts/evaluate_r2ad.py \
  --config configs/cf_dataset_epito.yaml --safety-area SafetyZone \
  --input /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --checkpoint results/r2ad/SafetyZone_p128_z64/model_r2ad.pt \
  --output-dir results/r2ad/SafetyZone_p128_z64/calibration \
  --calibrate --batch-size 32 --device cuda
```

This uses 128-pixel inference patches and stride 32 without repeating the size in the command. Recalibrate each new patch-size/model experiment; do not reuse a 224-pixel threshold. Existing thresholds made with an explicit different stride still require that same stride during evaluation.

## Then compare a larger latent space

### Camera zoom and translation, without rotation

For a camera that can zoom and shift slightly, use these training/animation options instead of rotation:

```bash
--aug-translate 0.005 --aug-rotate 0 \
--aug-zoom-min 1.0 --aug-zoom-max 1.03 \
--aug-contrast 0.05 --aug-brightness 4 \
--patch-size 128 --fixed-patch-size \
--samples-per-frame 6 --save-cp-k 1 --latent-audit-every 5
```

Camera zoom is a centered affine scaling of both image and mask, followed by translation. A factor of 1.03 means 3% zoom-in; 1.0 is unchanged, and factors below 1 zoom out. Rotation is exactly zero with `--aug-rotate 0`. At 1920×1080, the translation bound remains ±9.6 pixels horizontally and ±5.4 vertically. The zoom and translation are sampled independently per patch.

This zoom is independent of source patch-size jitter. `--fixed-patch-size` keeps the source crop at 128×128 while objects within it can become larger under camera zoom. The network still receives 128×128. Training animations use the same zoom-aware sampler, record the sampled zoom in JSON, and inherit range settings from updated checkpoint metadata unless overridden. They do not change checkpoint weights. Defaults are `--aug-zoom-min 1 --aug-zoom-max 1`, retaining previous behavior and random seeds for older runs; custom camera zoom currently uses the extracted-image route. Add the zoom flags to the existing training command; changing them during resume is recorded as a new augmentation condition.

### Full-frame and safety-area panels in the augmentation animation

Add `--full-frame` to `scripts/animate_r2ad_training.py` to append these views for every random augmented scene:

- Hann-weighted SafetyZone reconstruction composited over faded input outside the zone.
- Full-size fixed-scale Reds residual map, white outside SafetyZone.
- Combined SafetyZone residual overlay and isolated map.
- PLeft, PRight, RoboArm and ConvBelt maps with coverage and q99 scores.

All four original masks from `data.mask_types` are required. The exact zoom/translation/rotation matrix used for the image is also applied to each area mask with nearest-neighbor interpolation. The model, nominal source patch size and scene match the random-patch panel; full-frame reconstruction uses an additional fixed overlapping grid. Training itself still optimizes random patches and does not stitch full-frame maps.

Full-frame animations now default to `--layout landscape`: the random-patch view is on the left and full-frame/area panels are on the right. `--width 1024` is the width of each section, so the combined landscape GIF is 2048 pixels wide. `--layout portrait` retains the earlier stacked view.

An existing portrait GIF can be rearranged without running model inference again:

```bash
python scripts/reformat_r2ad_animation.py \
  --input reports/r2ad_zoom_translation_full_frame.gif \
  --output reports/r2ad_zoom_translation_full_frame_landscape.gif \
  --split-height 640
```

The 640-pixel boundary applies to the shown 1024-wide GIF made from 1920×1080 frames. When a companion JSON and its original source image are available, omit `--split-height` for automatic detection. Reflow retains timing, loop behavior and image content, and updates the directional header to left/right. The original GIF is preserved. A landscape copy of the local example was generated without recomputing reconstructions.

```bash
python scripts/animate_r2ad_training.py \
  --config configs/cf_dataset_epito.yaml \
  --train-csv /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --checkpoint results/r2ad/SafetyZone_fast_b32/checkpoints/model_r2ad_epoch_0016.pt \
  --output reports/r2ad_zoom_translation_full_frame.gif --device cuda \
  --aug-translate 0.005 --aug-rotate 0 --aug-zoom-min 1 --aug-zoom-max 1.03 \
  --aug-contrast 0.05 --aug-brightness 4 --patch-scale-min 1 --patch-scale-max 1 \
  --num-patches 6 --seed 42 --fps 2 --map-vmin 0 --map-vmax 0.5 \
  --full-frame --full-frame-scope safety --outside-dim 0.35 --batch-size 32
```

This inherits the checkpoint's source patch size; an older 224-pixel model does not become a 128-pixel model by disabling size jitter. Grid stride defaults from that size and can be set explicitly with `--stride`. SafetyZone grid inference is still more expensive than one random patch per GIF frame, so begin with six samples. For repeated zooms of one scene, add `--image /path/to/a/frame/in/train.csv`.

Inference is now restricted to SafetyZone; the earlier `--full-frame-scope all` option has been removed. Outside-zone reconstruction pixels are the original input dimmed by `--outside-dim` (default 0.35; 1 retains normal input brightness). They are not model predictions. All residual maps show white outside their displayed area; uncovered pixels inside the ROI remain gray and are excluded from scoring. Boundary patches still obey `--min-coverage`, but their out-of-zone pixels are neither blended into the displayed reconstruction nor scored. The JSON full-image entry reports frame coverage and a q99 over covered ROI pixels only. Area views restrict the same model's stitched map; they are not independent detectors or calibrated alarms. Regenerate older GIFs using the safety-only grid: reformatting an old GIF cannot change the inference that originally produced it.

The residual map blends smoothed per-patch errors, as in regular inference; it is not recomputed by subtracting the stitched RGB reconstruction from the original scene. All panels share `--map-vmin`/`--map-vmax`. Companion JSON records full-grid settings and combined/per-area coverage, mean error and q99 per sample; the final PNG preserves the last composite at GIF resolution. No extra raw full-frame dataset is copied or created.

### Comparing latent widths

After the validation audit, compare **64 versus 128 latent dimensions** with the same 128-pixel patches, data split, augmentation, epoch budget, seeds and scoring configuration. Replace `--latent-dims 64` above with `--latent-dims 128` and use `results/r2ad/SafetyZone_p128_z128/model_r2ad.pt`. This must start a new model: a 64-wide checkpoint cannot be resumed as 128-wide because encoder/decoder parameter shapes differ. The trainer now rejects that request explicitly rather than silently retaining the old width.

Do not change patch size and latent width in one experiment and attribute the outcome solely to capacity. Keep a p128/z64 control. Training retains the existing dimension-averaged KL convention for compatibility; report both total KL and mean KL/dimension when comparing widths. A larger model may reconstruct anomalies better too, so validate reconstruction and later labeled anomaly detection separately.

## Automatic diagnostics at the end of training

New runs enable `--latent-audit` by default, using the final epoch's normal validation forward pass. Outputs appear at `latent_audit/epoch_0020/` for a 20-epoch target. `--latent-audit-every 5` also audits epochs 5, 10, 15, etc.; `--no-latent-audit` disables it. `--latent-active-threshold` changes the heuristic cutoff. Checkpoints/summary JSON retain the latest diagnostic summary. At least two validation patches are required.

These automatic diagnostics describe the model at that epoch, not necessarily the best checkpoint. Forward precision matches validation (BF16 when configured); the standalone audit uses FP32 and can audit the selected best checkpoint separately. Small differences between the two are expected. They do not claim occupied volume or guarantee that the decoder makes use of every active encoder dimension.
