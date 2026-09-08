## Preprocess Dataset:

### Generate Masked Video

- Options
  - --max-frames 500
  - --save-every-n 5
  - --target-size 128
  - --process-to [frames, safety-areas ]
  - --image-format jpg
  - --stretch
  - --all

```bash
# Process UnExpected Videos
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 8_0 \
  --camera back_view \
  --generate-masked-video \
  --no-save-frames \
  --save-every-n 5 \
  --safety-areas PLeft PRight ConvBelt RoboArm \
  --progress
```

---

### Convert Rosbag to Video/ Frames/ Safety Areas

#### Rosbag to Video

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --process-to video \
  --progress \
  --max-frames 500 \
  --save-every-n 50

# Process data to VIDEO full

pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --process-to video \
  --progress

#--------------------------------------
base_path="/Users/rashid/data/DS/SR/v6/Jul27/extracted_frames"

for sid_path in "$base_path"/*; do
  if [[ -d "$sid_path" ]]; then

    s_id=$(basename "$sid_path")

    echo "======================================"
    echo "Processing scenario: $s_id"
    echo "Path: $sid_path"
    echo "======================================"

    pixi run python scripts/process_rosbags_to_dataset.py \
      --config configs/cf_dataset_mac.yaml \
      --scenario $s_id \
      --camera back_view \
      --process-to video \
      --progress \
      --max-frames 50 \

  else
    echo "Skipping: $sid_path (not a directory)"
  fi
done

```

---

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
```

---

#### Rosbag to Safety Areas

```bash
#  Rosbags to Safety Areas using Masks
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 9_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft PRight \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch \
  --max-frames 500 \
  --save-every-n 50
```

### RUN for all Data - Final

For Smoke test, use fewwer frames:

- --max-frames 500 \
- --save-every-n 50

```bash
# Preprocess Unexpected Data
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 8_0 \
  --camera back_view \
  --process-to safety-areas \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch \

```

```bash
for sid in 8_0 8_1 8_2 8_3 8_4 9_0 10_0 10_1 11_0 11_1 12_0 12_1 13_0 13_1 14_1 14_1 15_0 16_0 16_1; do
  echo "========================================="
  echo "Processing scenario: $sid"

  pixi run python scripts/process_rosbags_to_dataset.py \
    --config configs/cf_dataset_mac.yaml \
    --scenario "$sid" \
    --camera back_view \
    --process-to safety-areas \
    --progress \
    --target-size 128 \
    --image-format png \
    --stretch \
    --max-frames 10

  echo "========================================="
done
```

## Check Usefulness of Safety Areas based on Motion Analysis:

```bash
base_path="/Users/rashid/data/DS/SR/v6/Jul27/extracted_frames"

for sid_path in "$base_path"/*; do
  if [[ -d "$sid_path/back_view/processed" ]]; then
    echo "Processing: $sid_path"

    pixi run python scripts/analyze_safety_area_motion.py \
      "$sid_path" \
      --camera back_view
  else
    echo "Skipping: $sid_path (no back_view/processed directory)"
  fi
done

```

```bash
pixi run python scripts/analyze_safety_area_motion.py \
  --camera back_view
```
