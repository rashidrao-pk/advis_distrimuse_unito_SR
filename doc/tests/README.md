```bash
python -m pytest tests/ -v
```

```bash
python -m pytest \
  tests/test_input_data.py \
  tests/test_preprocessing.py \
  tests/test_anomaly_scores.py \
  -v
```

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

```bash
python scripts/test_model_inference.py \
  --config configs/cf_dataset_epito.yaml \
  --data_source training \
  --safety_area PLeft PRight RoboArm ConvBelt \
  --max_images 32
```

> Run on configured inference/testing data:

```bash
python scripts/test_model_inference.py \
  --config configs/cf_dataset_epito.yaml \
  --data_source testing \
  --safety_area PLeft \
  --max_images 100
```

```bash
python scripts/test_model_inference.py \
  --config configs/cf_dataset_mac.yaml \
  --safety_area PLeft \
  --data_source training \
  --device mps \
  --max_images 32
```
