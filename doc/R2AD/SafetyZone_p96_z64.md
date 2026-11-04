## Audit

```bash
python scripts/audit_r2ad_safetyzone.py \
  --config configs/cf_dataset_epito.yaml \
  --manifest /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --split train \
  --frame-index 0 \
  --patch-size 96 \
  --fixed-patch-size \
  --stride 96 \
  --min-coverage 0.85 \
  --samples-per-frame 6 \
  --simulate-frames 50 \
  --seed 42 \
  --output-dir results/r2ad/coverage_audit
```

---

# ------------------------------------------------------------------------------------------

## Create Animations

### For train

```bash
python scripts/animate_r2ad_training.py \
  --config configs/cf_dataset_epito.yaml \
  --train-csv /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --checkpoint results/r2ad/SafetyZone_p96_z64/checkpoints/model_r2ad_epoch_0016.pt \
  --output reports/r2ad_training_p96_z64_epoch18.gif \
  --device cuda \
  --patch-size 96 \
  --min-coverage 0.85 \
  --num-patches 24 \
  --seed 42 \
  --fps 3 \
  --map-vmin 0 \
  --map-vmax 0.5
```

### For Inference

```bash
python scripts/animate_r2ad_inference.py \
  --config configs/cf_dataset_epito.yaml \
  --image /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/sources/unexpected/scenario_8_1/frame_000164.png \
  --checkpoint results/r2ad/SafetyZone_p96_z64/checkpoints/model_r2ad_epoch_0018.pt \
  --output reports/r2ad_inference_p64_z64_epoch18.gif \
  --device cuda \
  --patch-size 96 \
  --stride 96 \
  --min-coverage 0.85 \
  --fps 3 \
  --map-vmin 0 \
  --map-vmax 0.5
```

## TRAIN MODELS

```bash
export R2AD_RUN=results/r2ad/SafetyZone_p96_z64
export R2AD_SPLIT=/beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split

python -u scripts/train_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --safety-area SafetyZone \
  --train-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --val-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --output results/r2ad/SafetyZone_p96_z64/model_r2ad.pt \
  --patch-size 96 \
  --fixed-patch-size \
  --latent-dims 64 \
  --epochs 20 \
  --batch-size 32 \
  --workers 8 \
  --device cuda \
  --precision bf16 \
  --aug-translate 0.005 \
  --aug-rotate 0 \
  --aug-zoom-min 1.0 \
  --aug-zoom-max 1.03 \
  --aug-contrast 0.05 \
  --aug-brightness 4 \
  --samples-per-frame 6 \
  --save-cp-k 1 \
  --latent-audit-every 5
```

## Inference

```bash
# SafetyZone_p96_z64
export R2AD_RUN=results/r2ad/SafetyZone_p96_z64
export R2AD_SPLIT=/beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split


python -u scripts/evaluate_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --safety-area SafetyZone \
  --input "$R2AD_SPLIT/val.csv" \
  --checkpoint "$R2AD_RUN/model_r2ad.pt" \
  --output-dir "$R2AD_RUN/calibration" \
  --calibrate --threshold-percentile 99 \
  --patch-size 96 --stride 96 \
  --min-coverage 0.85 --sigma 1 \
  --batch-size 32 --device cuda
```

#### Evaluate every test frame using that threshold:

```bash
python -u scripts/evaluate_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --safety-area SafetyZone \
  --input "$R2AD_SPLIT/test.csv" \
  --checkpoint "$R2AD_RUN/model_r2ad.pt" \
  --threshold-json "$R2AD_RUN/calibration/threshold.json" \
  --output-dir "$R2AD_RUN/test" \
  --patch-size 96 --stride 96 \
  --min-coverage 0.85 --sigma 1 \
  --batch-size 64 --device cuda
```

```bash
python scripts/inspect_r2ad_checkpoints.py \
  --config configs/cf_dataset_epito.yaml \
  --split-dir "$R2AD_SPLIT" \
  --checkpoints "$R2AD_RUN/model_r2ad.pt" \
  --output-dir "$R2AD_RUN/test_preview" \
  --safety-area SafetyZone \
  --samples-per-class 6 \
  --patch-size 96 --stride 96 \
  --device cuda \
  --map-vmin 0 --map-vmax 0.5
```

## Check Performance:

```bash
python scripts/benchmark_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --input /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/test.csv \
  --checkpoint results/r2ad/SafetyZone_p96_z64/model_r2ad.pt \
  --output-dir results/r2ad/SafetyZone_p96_z64/benchmark_b32 \
  --patch-size 96 --stride 96 \
  --min-coverage 0.85 --sigma 1 \
  --batch-size 32 \
  --device cuda:0 \
  --warmup 10 --frames 200
```
