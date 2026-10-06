# E1 — Spatial Risk Forecasting from ADVIS anomaly maps

## Hypothesis
The temporal evolution of ADVIS raw TAAS anomaly maps contains predictive information about where a future unexpected condition will emerge.

## Why this implementation matches the current ADVIS workflow
The live/offline pipeline runs one VAE-GAN per safety area (`PLeft`, `PRight`, `RoboArm`, `ConvBelt`). Each area is masked/cropped, resized to 128×128, reconstructed, then compared with TAAS to obtain a raw `distance` map. ADVIS converts the quantile of that map into a scalar anomaly score and compares it to a calibrated threshold.

E1 exports the **raw numeric TAAS map before colormap rendering**, then normalizes it exactly like the current dashboard logic:

`risk = clip(distance / (2 * effective_threshold), 0, 1)`

Thus `risk≈0.5` corresponds to the current ADVIS decision threshold. We do **not** train on rendered JPEG heatmaps.

Also, each 128×128 map is local to one safety-area crop. Therefore E1 treats one safety area as one temporal sample rather than concatenating the four maps as if their pixel coordinates were globally aligned.

## Files
- `export_anomaly_maps.py`: runs the existing VAE-GAN + TAAS path and saves raw/risk maps to NPZ.
- `dataset.py`: creates leak-free temporal windows within each scenario and safety area.
- `model.py`: shared per-area ConvLSTM forecaster.
- `train_e1.py`: trains future-map prediction.
- `evaluate_e1.py`: compares ConvLSTM against the mandatory persistence baseline.

## Recommended first protocol
Use complete scenarios as split units. Never randomly split frames/windows across train/test.

Example starting split (adjust after checking anomaly coverage):
- train: anomalous scenarios `8_0 8_1 8_2 9_0 10_0 11_0 12_0 13_0 14_0 15_0 16_0`
- val: `8_3 10_1 12_1 14_1`
- test: `8_4 11_1 13_1 16_1`

The goal of E1 is not final benchmark performance; it is to determine whether temporal anomaly-map dynamics beat persistence on **unseen complete scenarios**.

## 1. Export maps
Place this folder at `experiments/e1_spatial_risk/` in the ADVIS repo.

Example on Epito:

```bash
python experiments/e1_spatial_risk/export_anomaly_maps.py \
  --config configs/cf_dataset_epito.yaml \
  --scenario 13_1 \
  --threshold-calibration-mode val \
  --threshold-strategy percentile \
  --offset 3 \
  --sigma 2.0 \
  --quantile 0.99 \
  --threshold-amplification 1.5 1.15 1.2 1.2 \
  --taas-backend cython \
  --effective-fps 5
```

Run for each scenario. Output defaults to:

`results/V6/e1_spatial_risk/exports/scenario_<ID>_maps.npz`

NPZ shape:
- `risk_maps`: `[T, A, 128, 128]`
- `distance_maps`: `[T, A, 128, 128]`
- `scores`: `[T, A]`
- `thresholds`: `[T, A]`
- `areas`: area names
- `sample_ids`: source frame labels

## 2. Train
For a 5 fps exported stream, `history=15` means 3 s of context. The default horizon frames `(3,5,10,15)` correspond to approximately `(0.6,1,2,3)` seconds.

```bash
python experiments/e1_spatial_risk/train_e1.py \
  --train results/V6/e1_spatial_risk/exports/scenario_8_0_maps.npz \
          results/V6/e1_spatial_risk/exports/scenario_8_1_maps.npz \
          results/V6/e1_spatial_risk/exports/scenario_9_0_maps.npz \
  --val   results/V6/e1_spatial_risk/exports/scenario_8_3_maps.npz \
          results/V6/e1_spatial_risk/exports/scenario_12_1_maps.npz \
  --history 15 \
  --horizons 3 5 10 15 \
  --epochs 30 \
  --batch-size 8
```

## 3. Evaluate against persistence

```bash
python experiments/e1_spatial_risk/evaluate_e1.py \
  --checkpoint results/V6/e1_spatial_risk/convlstm_e1.pt \
  --test results/V6/e1_spatial_risk/exports/scenario_13_1_maps.npz \
         results/V6/e1_spatial_risk/exports/scenario_16_1_maps.npz \
  --history 15 \
  --horizons 3 5 10 15
```

E1 is considered promising only if ConvLSTM consistently improves over persistence, especially at +2 s and +3 s.

## Next metrics after the smoke test
Add per-horizon AUPRC, scenario/area breakdown, future anomaly-onset MAE, early-warning time, and spatial center error. For the first validity check, MAE + IoU vs persistence are intentionally kept simple.
