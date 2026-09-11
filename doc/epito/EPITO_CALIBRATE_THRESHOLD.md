```bash
ssh epito-mercurio

sinfo -N -p mirri,gracehopper,cascadelake,epito \
  -o "%.18N %.18P %.10T %.16G %.20C"

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

## Reconnect:

```bash
ssh epito-mercurio
squeue -u mrashid
# 419800  epito  ShapBPT Tests  RUNNING  epito02
srun --jobid=419800 --overlap --pty /bin/bash --noprofile --norc
tmux ls
tmux attach -t shapbpt
```

## Threshold with Max Value in Validation Data
```bash
# -------------------------------------
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

# python scripts/calibrate_threshold.py --mode val --safety_area PRight --dataset_version V6
python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PRight --dataset_version V6 --threshold_strategy max

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PLeft --dataset_version V6

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ConvBelt --dataset_version V6

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area RoboArm --dataset_version V6


#  Plot Validation Timeline
python scripts/plot_validation_timelines.py \
  --dataset_version V6 \
  --threshold_strategy max

```

---


## Threshold with Max Value in Validation Data
```bash
# -------------------------------------
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

# python scripts/calibrate_threshold.py --mode val --safety_area PRight --dataset_version V6
python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PRight --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PLeft --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ConvBelt --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area RoboArm --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0


python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ALL --dataset_version V6 --threshold_strategy percentile --threshold_percentile 99.0

-----------------
for strategy in max percentile; do
  python scripts/calibrate_threshold.py \
    --config configs/cf_dataset_epito.yaml \
    --mode val \
    --safety_area ALL \
    --dataset_version V6 \
    --threshold_strategy "$strategy"
done

#  Plot Validation Timeline
python scripts/plot_validation_timelines.py \
  --dataset_version V6 \
  --threshold_strategy percentile

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
  --max_images 1000
```