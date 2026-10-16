1. Process Dataset

````bash
#### Rosbag to Frames

```bash
# Process UnExpected Videos to frames
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 9_0 \
  --camera back_view \
  --process-to frames \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch \
  --max-frames 500 \
  --save-every-n 50
````

4. Run preprocessing

```bash
python scripts/preprocess_r2ad.py \
  --config configs/cf_dataset_mac.yaml \
  --mask-only

```

```bash
python scripts/preprocess_r2ad.py \
  --config configs/cf_dataset_mac.yaml \
  --val-fraction 0.20 \
  --test-fraction 0.20 \
  --split-mode grouped
```
