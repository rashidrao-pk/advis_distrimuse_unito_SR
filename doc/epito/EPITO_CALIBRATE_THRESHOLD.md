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

sinfo --format="%P %G %C"

squeue -u mrashid


srun --jobid=<JOB_ID> --pty bash

scancel 92873

sacct -j 92623 \
  --format=JobID,JobName%25,State,Elapsed,ExitCode,MaxRSS,NodeList
```

## Threshold with Max Value in Validation Data

```bash
# -------------------------------------
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

# # python scripts/calibrate_threshold.py --mode val --safety_area PRight --dataset_version V6
# python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PRight --dataset_version V6 --threshold_strategy max

# python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PLeft --dataset_version V6

# python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ConvBelt --dataset_version V6

# python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area RoboArm --dataset_version V6

```

---

## Threshold with Quantile Value in Validation Data

```bash
# -------------------------------------
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

# python scripts/calibrate_threshold.py --mode val --safety_area PRight --dataset_version V6
python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PRight --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ALL --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0


#  Plot Validation Timeline
python scripts/plot_validation_timelines.py \
  --dataset_version V6 \
  --threshold_strategy percentile

```

## Compute for Varios OFFSET, SIGMA, and QUANTILE Variants

```bash
#  OFFSET, SIGMA, and QUANTILE variants
for ooff in 1 2 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      python scripts/calibrate_threshold.py \
        --config configs/cf_dataset_epito.yaml \
        --mode val \
        --safety_area ALL \
        --dataset_version V6 \
        --threshold_strategy percentile \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" || exit 1
    done
  done
done
```

```bash
chmod +x scripts/bash/run_calibration.sh
./scripts/bash/run_calibration.sh
```

## Plot Validation Plots

```bash
for ooff in 1 2 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      echo "=============================================================================================="
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "=============================================================================================="
      python scripts/plot_validation_timelines.py \
        --dataset_version V6 \
        --threshold_strategy percentile \
        --threshold_percentile 99.0 \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" || exit 1
    done
  done
done
```

```bash
#  Plot Validation Timeline
python scripts/plot_validation_timelines.py \
  --dataset_version V6 \
  --threshold_strategy percentile
```

## Various Strategy

```bash
-----------------
# Varios strategy
for strategy in max percentile; do
  python scripts/calibrate_threshold.py \
    --config configs/cf_dataset_epito.yaml \
    --mode val \
    --safety_area ALL \
    --dataset_version V6 \
    --threshold_strategy "$strategy" \
done
```

## Ablation Analysis:

```bash
python scripts/ablate_validation_thresholds.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area PLeft \
  --dataset_version V6
```

```bash
python scripts/ablate_validation_thresholds.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area PLeft \
  --dataset_version V6 \
  --offsets 0,1,2 \
  --sigmas 0,0.5,1.0 \
  --quantiles 0.95,0.99,0.999 \
  --threshold_percentiles 95,99,99.5 \
  --threshold_n_sigmas 2,3 \
  --target_normal_fpr 0.01
```

```bash
python scripts/ablate_validation_thresholds.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area PLeft \
  --dataset_version V6 \
<<<<<<< Updated upstream
  --max_images 1000
```

=======
--max_images 200

```bash
for SafetyArea in PLeft PRight ConvBelt RoboArm; do
python scripts/ablate_validation_thresholds.py \
 --config configs/cf_dataset_epito.yaml \
 --safety_area "$SafetyArea" \
 --dataset_version V6
done

```

## Reconnect:

```bash
ssh epito-mercurio
squeue -u mrashid
# 419800  epito  ShapBPT Tests  RUNNING  epito02
srun --jobid=419800 --overlap --pty /bin/bash --noprofile --norc
tmux ls
tmux attach -t shapbpt
```

> > > > > > > Stashed changes

## After New TAAS Variant:

- `--taas_backend` `numpy` | `cython`
- `--taas_variant` `canonical` | `minimization`

```bash
python scripts/calibrate_threshold.py \
  --config configs/cf_dataset_epito.yaml \
  --dataset_version V6 \
  --mode val \
  --safety_area ALL \
  --threshold_strategy percentile \
  --threshold_percentile 99.0 \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --taas_backend cython
```

```bash
set -euo pipefail

# Build the Cython extension for the active Python environment.
python scripts/setup_taas_cython.py build_ext --inplace --force

total=18
current=0

for ooff in 1 2 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))

      echo "============================================================"
      echo "Calibration ${current}/${total}"
      echo "TAAS variant=canonical"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "============================================================"

      python scripts/calibrate_threshold.py \
        --config configs/cf_dataset_epito.yaml \
        --mode val \
        --safety_area ALL \
        --dataset_version V6 \
        --threshold_strategy percentile \
        --threshold_percentile 99.0 \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" \
        --taas_backend cython \
        --taas_variant canonical
    done
  done
done
echo "[COMPLETED] All ${total} canonical TAAS calibrations"
```

## Threshold with TEST DATA:

```bash
# -------------------------------------
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

python scripts/calibrate_threshold.py \
  --mode test \
  --safety_area ALL \


# python scripts/calibrate_threshold.py --mode val --safety_area PRight --dataset_version V6

# python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ALL --gt_csv scripts/data/annotations.csv
python scripts/calibrate_threshold.py \
  --config configs/cf_dataset_epito.yaml \
  --mode test \
  --dataset_version V6 \
  --safety_area ALL \
  --test_folder /Users/rashid/data/DS/SR/v6/Jul27/test \
  --test_scenarios 13_0 \
  --gt_csv reports/safety_area_annotations/saved_annotation/scenario_13_0_back_view_annotations.csv \
  --camera back_view \
  --threshold_method f1c \
  --threshold_strategy percentile \
  --threshold_percentile 99.0 \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --taas_backend cython


python scripts/calibrate_threshold.py \
  --config configs/cf_dataset_epito.yaml \
  --mode test \
  --dataset_version V6 \
  --safety_area ALL \
  --test_folder /beegfs/home/mrashid/datasets/AD/SR/V6/test \
  --test_scenarios 13_0 \
  --gt_csv reports/safety_area_annotations/saved_annotation/scenario_13_0_back_view_annotations.csv \
  --camera back_view \
  --threshold_method f1c \
  --threshold_strategy percentile \
  --threshold_percentile 99.0 \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --taas_backend cython

# --offset_ls 1,2,3 \
# --sigma_ls 1.0,1.5 \
# --quantile_ls 0.97,0.98,0.99 \

#  Plot Validation Timeline
python scripts/plot_validation_timelines.py \
  --dataset_version V6 \
  --threshold_strategy percentile
```

```bash
python scripts/calibrate_threshold.py \
  --config configs/cf_dataset_epito.yaml \
  --mode test \
  --dataset_version V6 \
  --safety_area ALL \
  --test_folder /beegfs/home/mrashid/datasets/AD/SR/V6/test \
  --test_scenarios 8_16 \
  --gt_csv reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  --camera back_view \
  --threshold_method f1c \
  --taas_backend auto
```
