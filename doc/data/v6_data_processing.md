## Creat a single Unified Anomalous Video from Already Videos

Options:

- Generate everything again from rosbags
  - Use `--force`:

- To include them with grey Unlabeled masks
  - `--missing-annotations Unlabeled` or `--missing-annotations skip` or `error`

```bash
pixi run python scripts/create_unified_anomalous_video.py \
  --base-path /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames \
  --config configs/cf_dataset_mac.yaml \
  --after 8_0 \
  --camera front_view \
  --progress
```

```bash
pixi run python scripts/create_unified_anomalous_video.py \
  --base-path /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames \
  --config configs/cf_dataset_mac.yaml \
  --after 8_0 \
  --camera back_view \
  --progress
```

```bash
pixi run python scripts/create_unified_anomalous_video.py \
  --base-path /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames \
  --config configs/cf_dataset_mac.yaml \
  --after 8_0 \
  --camera back_view \
  --annotated-masks \
  --annotations-dir reports/safety_area_annotations/saved_annotation \
  --areas PLeft PRight RoboArm ConvBelt \
  --missing-annotations skip \
  --progress
```
