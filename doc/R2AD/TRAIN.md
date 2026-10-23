```bash
python scripts/train_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --safety-area SafetyZone \
  --train-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --val-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --output results/r2ad/SafetyZone_run01/smoke/model_r2ad.pt \
  --patch-size 224 \
  --samples-per-frame 6 \
  --epochs 1 \
  --batch-size 32 \
  --workers 4 \
  --monitor-every 1000 \
  --device cuda
```

### Improved Training

- BF16 automatically on supported CUDA GPUs, with loss calculations kept in FP32.
- Patch-only image augmentation, avoiding full-frame image transforms for each patch.
- Cached masks, persistent loader workers, and controlled OpenCV threading.
- Monitoring every 1,000 steps by default, incremental CSV writes, and cheaper loss plots.
- Warm-up-aware ETA, patches/second, and data-loader wait measurements.

```bash
python -u scripts/train_r2ad.py \
  --config configs/cf_dataset_epito.yaml \
  --safety-area SafetyZone \
  --train-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv \
  --val-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv \
  --output results/r2ad/SafetyZone_fast_b32/model_r2ad.pt \
  --patch-size 224 \
  --samples-per-frame 6 \
  --epochs 1 \
  --batch-size 32 \
  --workers 8 \
  --device cuda \
  --precision bf16 \
  --augmentation-mode crop \
  --persistent-workers \
  --prefetch-factor 2 \
  --monitor-every 1000
```
