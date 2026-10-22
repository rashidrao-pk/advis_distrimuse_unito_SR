# Process Dataset

## 2. Run for one scenario

```bash
python scripts/process_rosbags_to_frames_rrad.py \
  --config configs/cf_dataset_epito.yaml \
  --scenario 1_0 \
  --camera back_view \
  --process-to frames \
  --progress \
  --max-frames 50
```

## 3. Run for all scenarios

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
  --split-mode grouped
```

```bash
python scripts/preprocess_r2ad.py \
  --config configs/cf_dataset_mac.yaml \
  --val-fraction 0.20 \
  --test-fraction 0.20 \
  --split-mode temporal
```
