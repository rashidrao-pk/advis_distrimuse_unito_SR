### Inspect without saving

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --dry-run
```

### Extract the selected YAML scenario and configured back camera:

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml
```

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --camera back_view front_view
```

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 3_0 3_1 \
  --camera back_view \
  --save-every-n 10
```

### All Scenarios

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --all \
  --camera back_view front_view \
  --save-every-n 10
```

## Generate Masked Video

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --generate-masked-video \
  --no-save-frames \
  --safety-areas PLeft PRight\
  --progress \
  --max-frames 500 \
```

```bash
pixi run python scripts/process_rosbags_from_config.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --generate-masked-video \
  --no-save-frames \
  --save-every-n 10 \
  --safety-areas PLeft PRight ConvBelt RoboArm \
  --progress \
  --max-frames 500 \
```

## Preprocess Dataset:

- Options
  - --max-frames 500
  - --save-every-n 5
  - --target-size 128
  - --image-format jpg
  - --stretch
  - --all

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft PRight \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch \
  --max-frames 500
```

### RUN for all Data - Final

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 1_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PRight \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch
```

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 2_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch
```

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 2_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch
```

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 3_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PRight ConvBelt RoboArm \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch

pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 4_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft ConvBelt RoboArm \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch

pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 5_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PRight RoboArm \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch

pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 6_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft PRight RoboArm \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch


pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac.yaml \
  --scenario 7_0 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PLeft PRight RoboArm ConvBelt\
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch

```

## SAVE to SSD

```bash
pixi run python scripts/process_rosbags_to_dataset.py \
  --config configs/cf_dataset_mac_SSD_ext.yaml \
  --scenario 3_1 \
  --camera back_view \
  --process-to safety-areas \
  --safety-areas PRight ConvBelt RoboArm \
  --progress \
  --target-size 128 \
  --image-format png \
  --stretch
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
