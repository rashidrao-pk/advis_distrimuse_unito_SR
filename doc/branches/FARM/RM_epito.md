## RUN on

- epito

```bash
ssh epito-mercurio

# Check all Available sources
sinfo -N -p mirri,gracehopper,cascadelake,epito \
  -o "%.18N %.18P %.10T %.16G %.20C"

# Check all Reserved sources
squeue -p mirri,gracehopper,cascadelake,epito \
  -o "%.12i %.12u %.18P %.18j %.8T %.15N %.12b %.20R"

srun -p epito --gres=gpu:a100:1 -J "ADVIS-Cal" --pty bash
tmux new -s AD_SR_Cal
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR
```

```bash
python experiments/e1_spatial_risk/export_anomaly_maps.py \
  --config configs/cf_dataset_epito.yaml \
  --scenario 13_1 \
  --threshold-calibration-mode val \
  --threshold-strategy percentile \
  --offset 3 \
  --sigma 2.0 \
  --quantile 0.99 \
  --threshold-amplification 1.5 1.15 1.2 1.2 \
  --taas-backend cython \
  --effective-fps 5
```

```bash
python - <<'PY'
import numpy as np

p = "results/V6/e1_spatial_risk/exports/scenario_13_1_maps.npz"
d = np.load(p, allow_pickle=True)

print("Keys:", d.files)
for k in d.files:
    x = d[k]
    print(f"{k:15s} shape={x.shape} dtype={x.dtype}")

print("\nAreas:", d["areas"])

risk = d["risk_maps"]
scores = d["scores"]

print("\nRisk statistics")
print("min :", risk.min())
print("mean:", risk.mean())
print("max :", risk.max())

print("\nScore statistics by area")
for i, area in enumerate(d["areas"]):
    print(
        area,
        "min=", scores[:, i].min(),
        "mean=", scores[:, i].mean(),
        "max=", scores[:, i].max(),
    )
PY
```

## Run for All Scenarios

```bash
for sid in \
8_0 8_1 8_2 \
8_3 8_4 \
9_0 \
10_0 10_1 \
11_0 11_1 \
12_0 12_1 \
13_0 \
14_0 14_1 \
15_0 \
16_0 16_1
do

  echo "======================================"
  echo "E1 export scenario: $sid"
  echo "======================================"

  python experiments/e1_spatial_risk/export_anomaly_maps.py \
    --config configs/cf_dataset_epito.yaml \
    --scenario "$sid" \
    --threshold-calibration-mode val \
    --threshold-strategy percentile \
    --offset 3 \
    --sigma 2.0 \
    --quantile 0.99 \
    --threshold-amplification 1.5 1.15 1.2 1.2 \
    --taas-backend cython \
    --effective-fps 5

done
```

```bash
python - <<'PY'
import numpy as np

p = "results/V6/e1_spatial_risk/exports/scenario_13_1_maps.npz"
d = np.load(p, allow_pickle=True)

scores = d["scores"]
thresholds = d["thresholds"]
areas = d["areas"]

norm = scores / thresholds

print("\n=== E1.0 threshold crossing analysis ===")

for i, area in enumerate(areas):
    x = norm[:, i]

    anomalous = x > 1.0
    idx = np.where(anomalous)[0]

    print(f"\n{area}")
    print(f"  min norm : {x.min():.3f}")
    print(f"  mean norm: {x.mean():.3f}")
    print(f"  max norm : {x.max():.3f}")
    print(f"  anomalous frames: {anomalous.sum()} / {len(x)} "
          f"({100*anomalous.mean():.1f}%)")

    if len(idx):
        print(f"  first threshold crossing: frame {idx[0]}")
        print(f"  last threshold crossing : frame {idx[-1]}")

        # contiguous anomaly starts
        starts = np.where(
            anomalous & np.r_[True, ~anomalous[:-1]]
        )[0]

        print(f"  anomaly episode starts: {starts[:20]}")
    else:
        print("  no threshold crossing")
PY
```

### PLOT

```bash
python experiments/e1_spatial_risk/plot_e1_precursors.py \
  --input-dir results/V6/e1_spatial_risk/exports \
  --fps 5

open results/V6/e1_spatial_risk/precursor_plots
```

```bash
ls -lh results/V6/e1_spatial_risk/exports/
```

## TRAIN:

### Smoke test

```bash
python experiments/e1_spatial_risk/train_e1.py \
  --train \
    results/V6/e1_spatial_risk/exports/scenario_8_0_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_8_1_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_13_0_maps.npz \
  --val \
    results/V6/e1_spatial_risk/exports/scenario_8_3_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_12_1_maps.npz \
  --history 15 \
  --horizons 3 5 10 15 \
  --epochs 2 \
  --batch-size 4 \
  --num-workers 0
```

```bash
python experiments/e1_spatial_risk/train_e1.py \
  --train \
    results/V6/e1_spatial_risk/exports/scenario_8_0_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_8_1_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_8_2_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_9_0_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_12_0_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_15_0_maps.npz \
  --val \
    results/V6/e1_spatial_risk/exports/scenario_8_3_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_12_1_maps.npz \
  --history 15 \
  --horizons 3 5 10 15 \
  --window-stride 1 \
  --hidden 32 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001 \
  --num-workers 4 \
  --output results/V6/e1_spatial_risk/e1A_convlstm.pt
```

## Evaluate Model

```bash
python experiments/e1_spatial_risk/evaluate_e1.py \
  --checkpoint results/V6/e1_spatial_risk/e1A_convlstm.pt \
  --test \
    results/V6/e1_spatial_risk/exports/scenario_8_4_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_16_0_maps.npz \
  --history 15 \
  --horizons 3 5 10 15 \
  --batch-size 32
```

```bash
python experiments/e1_spatial_risk/evaluate_e1.py \
  --checkpoint results/V6/e1_spatial_risk/e1A_convlstm.pt \
  --test \
    results/V6/e1_spatial_risk/exports/scenario_8_4_maps.npz \
    results/V6/e1_spatial_risk/exports/scenario_16_0_maps.npz \
  --history 15 \
  --horizons 3 5 10 15 \
  --batch-size 32 \
  --num-workers 4 \
  --fps 5 \
  --score-threshold 1.0 \
  --min-anomaly-frames 3 \
  --trend-window 5 \
  --output-dir results/V6/e1_spatial_risk/eval_e1B
```
