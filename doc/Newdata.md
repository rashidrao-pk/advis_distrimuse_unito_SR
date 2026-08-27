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
