# SafetyZone run: epoch-16 review (2026-10-25)

Reviewed `results/r2ad/SafetyZone_fast_b32/training_progress.png`, `epoch_metrics.csv`, `live_monitor/probe_epoch_016.png`, and representative first/middle/final frames of `reports/r2ad_training_epoch16.gif` and `reports/r2ad_inference_epoch16.gif`. The progress plot already includes a partial epoch 17, while the CSV contains 16 completed epochs. The GIF companion JSON files were not present locally, so augmentation settings are established from the sampler code rather than per-frame animation provenance.

## Evidence

| Metric | Epoch 1 | Epoch 15 | Epoch 16 |
|---|---:|---:|---:|
| Training reconstruction MSE | 0.122346 | 0.068209 | 0.067492 |
| Validation reconstruction MSE | 0.106037 | **0.080698** | 0.083616 |
| Generator adversarial loss | 2.3687 | 5.4334 | 5.5861 |
| Discriminator BCE | 0.26652 | 0.04169 | 0.03720 |
| Discriminator accuracy | 89.31% | 98.58% | 98.73% |

The model is learning normal appearance, with training MSE down about 45% and the best validation MSE down about 24%. Epoch 15 is the best validation checkpoint through epoch 16. The epoch-16 increase is about 3.6% relative to epoch 15, but one fluctuation with resampled validation patches is insufficient to establish overfitting.

The fixed probes preserve major box/conveyor/floor structure but show deformed boundaries, lost detail and strong residuals on normal edges. These validation probes are not geometrically/photometrically augmented, so the displayed distortions are model reconstruction errors, not simply the augmentation being displayed. Random training patches also reconstruct the broad scene while losing fine structure. The inference GIF shows substantial residual on normal conveyor/box structure as well as the scene's anomalous content; it does not demonstrate reliable localization of anomalies.

The near-perfect real/reconstruction discriminator score and increasing adversarial generator loss indicate adversarial imbalance. This, the 128×128 network input, latent bottleneck and stochastic VAE training are alternative contributors to reconstruction limitations. The supplied artifacts alone cannot isolate augmentation as the cause. Training reconstructs the augmented patch itself; it does not try to undo augmentation to match an untransformed target.

## What the existing augmentation actually does

| Control | Previous/default range | Gentle controlled trial |
|---|---|---|
| Translation | ±3% of full image size: ±57.6 px x, ±32.4 px y at 1920×1080 | ±0.5%: ±9.6 px x, ±5.4 px y |
| Rotation | ±2° | ±0.5° |
| Contrast multiplier | 0.85–1.15 | 0.95–1.05 |
| Brightness offset | ±12 pixel values | ±4 |
| Source patch side | 0.9–1.1 × 224 ≈ 202–246 px | 0.98–1.02 × 224 ≈ 220–228 px |
| Affine zoom | None; transform scale is 1.0 | Still none |

Apparent zoom comes from different crop sizes being resized to the same 128×128 network input. Random patch centers and changes between source frames also make the GIF jump; reducing translation does not turn random patch selection into a smooth camera trajectory.

## New controls

Both `train_r2ad.py` (image datasets) and `animate_r2ad_training.py` now accept:

```bash
--aug-translate 0.005 --aug-rotate 0.5 \
--aug-contrast 0.05 --aug-brightness 4 \
--patch-scale-min 0.98 --patch-scale-max 1.02
```

Defaults retain the previous behavior. Set geometric/photometric jitter values to 0 and both patch scales to 1 for an ablation with fixed crop size and no augmentation (centers remain random). Animation inherits these range settings from updated checkpoint metadata unless explicitly overridden. A preview override changes visualization inputs, not trained weights.

For a controlled continuation, preserve this run and use a new output directory, keeping model/objective/learning-rate settings unchanged:

```bash
python -u scripts/train_r2ad.py \
  --resume results/r2ad/SafetyZone_fast_b32/checkpoints/model_r2ad_epoch_0016.pt \
  --output results/r2ad/SafetyZone_gentle_from16/model_r2ad.pt \
  --epochs 20 --device cuda --save-cp-k 1 \
  --aug-translate 0.005 --aug-rotate 0.5 \
  --aug-contrast 0.05 --aug-brightness 4 \
  --patch-scale-min 0.98 --patch-scale-max 1.02
```

`--epochs 20` is the total target, giving four more epochs after a completed epoch 16. Old checkpoints provide a weights-only warm start with fresh optimizers; updated full-state checkpoints retain them. Changing augmentation is explicitly logged as a new training condition, not exact replay. Validation settings and fixed probes remain unchanged for comparison. Full-state resumes keep inherited best-model history; the new directory always receives `last_checkpoint.pt`, while its best-model output is written on a new validation improvement. Keep the prior best checkpoint separately.

Compare fixed-probe reconstruction, validation MSE over several epochs, and normal validation residuals. Do not alter GAN weights simultaneously in the first augmentation experiment, or claim causal improvement from one stochastic validation result. If adversarial imbalance remains, investigate it as a separate validation-driven experiment. Do not tune these controls on test-set scores.
