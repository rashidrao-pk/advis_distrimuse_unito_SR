```bash
python scripts/check_model_checkpoints.py \
 --config configs/cf_dataset_epito.yaml
```

```bash
python scripts/check_model_checkpoints.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area PLeft PRight
```

```bash
python scripts/check_model_checkpoints.py \
  --config configs/cf_dataset_epito.yaml \
  --json
```

```bash
python -m pytest tests/test_model_checkpoints.py -v
```
