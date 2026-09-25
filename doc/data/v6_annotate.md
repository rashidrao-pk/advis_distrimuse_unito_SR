## Annotate Safety Areas

Features included:

- Frame slider and Previous/Next controls
- Automatic playback at selectable speeds
- Safety-area selector
- Raw and processed frame views when raw frames exist
- Label a single frame or an inclusive frame sequence
- Normal, Anomalous, and Unlabeled states
- Optional annotation notes
- Keyboard shortcuts:
  - Left/Right arrows: navigate
  - N: label Normal
  - A: label Anomalous
- Browser autosave using local storage
- Import an existing annotation CSV
- Download a full-scenario CSV containing every frame
- Scenario description loaded from the YAML
- Dataset images referenced by path rather than embedded in the HTML

```bash
pixi run python scripts/annotate_safety_area.py \
  /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames/1_0 \
  --camera back_view
```

```bash
base_path="/Users/rashid/data/DS/SR/v6/Jul27/extracted_frames"

for sid_path in "$base_path"/*; do
  [[ -d "$sid_path" ]] || continue

  s_id=$(basename "$sid_path")

  # Accept IDs such as 9_0, 10_0, 13_1.
  if [[ "$s_id" =~ ^([0-9]+)_([0-9]+)$ ]]; then
    scenario_number="${match[1]}"

    # Skip scenarios 8_x and earlier.
    (( scenario_number > 8 )) || continue

    echo "======================================"
    echo "Processing anomalous scenario: $s_id"
    echo "Path: $sid_path"
    echo "======================================"

    pixi run python scripts/annotate_safety_area.py \
      "$sid_path" \
      --camera back_view \
      --areas PLeft PRight RoboArm ConvBelt
  fi
done
```

```bash
base_path="/Users/rashid/data/DS/SR/v6/Jul27/extracted_frames"

for sid in \
  8_0 8_1 8_2 8_3 8_4 \
  9_0 \
  10_0 10_1 \
  11_0 11_1 \
  12_0 12_1 \
  13_0 13_1 \
  14_0 14_1 \
  15_0 \
  16_0 16_1
do
  sid_path="$base_path/$sid"

  # Skip if the scenario directory does not exist
  if [[ ! -d "$sid_path" ]]; then
    echo "Skipping $sid: directory not found"
    continue
  fi

  echo "========================================="
  echo "Processing scenario: $sid"
  echo "Path: $sid_path"
  echo "========================================="

  # 1. Annotate safety areas
  pixi run python scripts/annotate_safety_area.py \
    "$sid_path" \
    --camera back_view \
    --areas PLeft PRight RoboArm ConvBelt

  echo "Finished: $sid"
  echo "========================================="
done
```

## PROCESS DATA

- Create normal and anomalous frames based on

```bash
pixi run python scripts/annotation_to_test_data.py --dry-run

## PROCESS FOR SELECTED SCENARIO
pixi run python scripts/annotation_to_test_data.py \
    --scenarios 8_16 \
    --progress

# PROCESS FOR ALL
for sid in \
  8_0 8_1 8_2 8_3 8_4 \
  9_0 \
  10_0 10_1 \
  11_0 11_1 \
  12_0 12_1 \
  13_0 13_1 \
  14_0 14_1 \
  15_0 \
  16_0 16_1
do

  pixi run python scripts/annotation_to_test_data.py \
    --scenarios "$sid" \
    --progress
done

pixi run python scripts/annotation_to_test_data.py \
  --config configs/cf_dataset_mac.yaml \
  --progress
```

### Create Test Data from Annotation (`All Scenarios togther`):

```bash
pixi run python scripts/annotation_to_test_data.py \
  --config configs/cf_dataset_mac.yaml \
  --annotation-csv reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  --camera back_view \
  --safety-areas PLeft PRight RoboArm ConvBelt \
  --mode copy \
  --progress
```
