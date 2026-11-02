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

| Diagnostic | Result |
|---|---:|
| Active dimensions at threshold 0.01 | 64 / 64 |
| Participation-ratio effective rank | 20.14 |
| Entropy effective rank | 31.40 |
| Components for 95% variation | 41 |
| Mean KL per dimension | 5.0546 nats |
| Independent source frames used | **1** |

Artifacts: `reports/r2ad_latent_audit_epoch12_exploratory/`. This is not the latest Epito model, not a held-out validation audit, and not evidence that 64 dimensions are saturated. All units are varying on this sample, but some directions are correlated. It does not establish that increasing width will help.

## Smaller source patches with matching inference

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

### Latent-width comparison

After the validation audit, compare **64 versus 128 latent dimensions** with the same 128-pixel patches, data split, augmentation, epoch budget, seeds and scoring configuration. Replace `--latent-dims 64` above with `--latent-dims 128` and use `results/r2ad/SafetyZone_p128_z128/model_r2ad.pt`. This must start a new model: a 64-wide checkpoint cannot be resumed as 128-wide because encoder/decoder parameter shapes differ. The trainer now rejects that request explicitly rather than silently retaining the old width.

Do not change patch size and latent width in one experiment and attribute the outcome solely to capacity. Keep a p128/z64 control. Training retains the existing dimension-averaged KL convention for compatibility; report both total KL and mean KL/dimension when comparing widths. A larger model may reconstruct anomalies better too, so validate reconstruction and later labeled anomaly detection separately.

## Automatic diagnostics at the end of training

New runs enable `--latent-audit` by default, using the final epoch's normal validation forward pass. Outputs appear at `latent_audit/epoch_0020/` for a 20-epoch target. `--latent-audit-every 5` also audits epochs 5, 10, 15, etc.; `--no-latent-audit` disables it. `--latent-active-threshold` changes the heuristic cutoff. Checkpoints/summary JSON retain the latest diagnostic summary. At least two validation patches are required.

These automatic diagnostics describe the model at that epoch, not necessarily the best checkpoint. Forward precision matches validation (BF16 when configured); the standalone audit uses FP32 and can audit the selected best checkpoint separately. Small differences between the two are expected. They do not claim occupied volume or guarantee that the decoder makes use of every active encoder dimension.
