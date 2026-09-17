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
