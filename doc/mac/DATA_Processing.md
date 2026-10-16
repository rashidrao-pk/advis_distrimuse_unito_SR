# Upload Data to Epito

````bash
zip /Users/rashid/data/DS/SR/v6/Jul27/Jul27_Scenario_1_0_2026-07-27_10-25-24

scp /Users/rashid/data/DS/SR/v6/Jul27/Jul27_Scenario_1_0_2026-07-27_10-25-24

# zip /Users/rashid/data/DS/SR/v6/Jul27/test

unzip -o /beegfs/home/mrashid/datasets/AD/SR/V6/test.zip \
  -d /beegfs/home/mrashid/datasets/AD/SR/V6/test \
  -x "__MACOSX/*"


scp /Users/rashid/data/DS/SR/v6/Jul27/Jul27_Scenario_1_0_2026-07-27_10-25-24 mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets/AD/SR/V6/rosbags

# Process Dataset

## Rosbag to Frames

```bash
# Process UnExpected Videos to frames
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 9_0 \
  --camera back_view \
  --process-to frames \
  --progress \
  --image-format png \
  --max-frames 50
````

## 2. Run for one scenario

```bash
pixi run python scripts/process_rosbags_to_frames_rrad.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --process-to frames \
  --progress \
  --max-frames 50
```

3. Run for all scenarios

```bash
pixi run python scripts/process_rosbags_to_frames_rrad.py \
  --config configs/cf_dataset_mac.yaml \
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
