## Create Animations

### For train

```bash
python scripts/animate_r2ad_training.py \
  --config configs/cf_dataset_epito.yaml \
  --train-csv /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --checkpoint results/r2ad/SafetyZone_p64_z64/checkpoints/model_r2ad_epoch_0016.pt \
  --output reports/r2ad_training_p64_z64_epoch18.gif \
  --device cuda \
  --patch-size 64 \
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
  --checkpoint results/r2ad/SafetyZone_p64_z64/checkpoints/model_r2ad_epoch_0018.pt \
  --output reports/r2ad_inference_p64_z64_epoch18.gif \
  --device cuda \
  --patch-size 64 \
  --stride 64 \
  --min-coverage 0.85 \
  --fps 3 \
  --map-vmin 0 \
  --map-vmax 0.5
```

## Inference

```bash
# SafetyZone_p64_z64
export R2AD_RUN=results/r2ad/SafetyZone_p64_z64
export R2AD_SPLIT=/beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split


python -u scripts/evaluate_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --safety-area SafetyZone \
  --input "$R2AD_SPLIT/val.csv" \
  --checkpoint "$R2AD_RUN/model_r2ad.pt" \
  --output-dir "$R2AD_RUN/calibration" \
  --calibrate --threshold-percentile 99 \
  --patch-size 64 --stride 64 \
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
  --patch-size 64 --stride 64 \
  --min-coverage 0.85 --sigma 1 \
  --batch-size 32 --device cuda
```

```bash
python scripts/inspect_r2ad_checkpoints.py \
  --config configs/cf_dataset_epito.yaml \
  --split-dir "$R2AD_SPLIT" \
  --checkpoints "$R2AD_RUN/model_r2ad.pt" \
  --output-dir "$R2AD_RUN/test_preview" \
  --safety-area SafetyZone \
  --samples-per-class 6 \
  --patch-size 64 --stride 64 \
  --device cuda \
  --map-vmin 0 --map-vmax 0.5
```
