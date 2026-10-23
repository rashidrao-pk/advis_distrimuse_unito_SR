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
  --batch-size 8 \
  --workers 4 \
  --device cuda
```
