```bash
# Run one-class training with:
python scripts/train_new.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area ConvBelt \
  --epochs 200 \
  --save_figures \
  --estimate_time \
  --batch_size 128
```

```bash
python3 scripts/train_new.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area ConvBelt \
  --batch_size 128 \
  --save_figures
```


