# Process Dataset

For the full current workflow and completion checks, use the [training runbook](TRAINING_RUNBOOK.md). The `--max-frames 50` examples below are extraction smoke tests only.

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

This writes virtual split CSVs without duplicating frames. Set `r2ad.train_frames`,
`val_frames`, `test_frames`, and `normal_test_frames` to the generated `train.csv`,
`val.csv`, `test.csv`, and `normal_test.csv` paths printed by the command.
Use a fresh `--output-dir` if the previous split directory is already populated.
