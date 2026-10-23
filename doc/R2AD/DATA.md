# Process Dataset

## 1. Run for all scenarios

```bash
python scripts/process_rosbags_to_frames_rrad.py \
  --config configs/cf_dataset_epito.yaml \
  --all \
  --camera back_view \
  --progress \
  --max-frames 50
```

```bash
python scripts/preprocess_r2ad.py \
  --config configs/cf_dataset_mac.yaml \
  --val-fraction 0.20 \
  --test-fraction 0.20 \
  --transfer manifest \
  --split-mode grouped
```

```bash
python scripts/preprocess_r2ad.py \
  --config configs/cf_dataset_mac.yaml \
  --val-fraction 0.20 \
  --test-fraction 0.20 \
  --transfer manifest \
  --split-mode temporal
```

## Preview the processed data

```bash
python scripts/preview_r2ad_splits.py \
  --config configs/cf_dataset_epito.yaml \
  --split-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split \
  --output-dir reports/r2ad_split_preview \
  --samples 6 \
  --seed 42
```
