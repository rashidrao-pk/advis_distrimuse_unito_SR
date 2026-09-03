```bash

sinfo -N -p mirri,gracehopper,cascadelake,epito \
  -o "%.18N %.18P %.10T %.16G %.20C"

squeue -p mirri,gracehopper,cascadelake,epito \
  -o "%.12i %.12u %.18P %.18j %.8T %.15N %.12b %.20R"

srun -p epito --gres=gpu:a100:1 -J "ADVIS Threshold Calibration" --pty bash
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
ssh epito
squeue -u mrashid
# 419800  epito  ShapBPT Tests  RUNNING  epito02
srun --jobid=419800 --overlap --pty /bin/bash --noprofile --norc
tmux ls
tmux attach -t shapbpt
```

```bash
# -------------------------------------
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

# python scripts/calibrate_threshold.py --mode val --safety_area PRight --dataset_version V6
python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PRight --dataset_version V6

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area PLeft --dataset_version V6

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area ConvBelt --dataset_version V6

python scripts/calibrate_threshold.py --config configs/cf_dataset_epito.yaml --mode val --safety_area RoboArm --dataset_version V6

#  Plot Validation Timeline
python scripts/plot_validation_timelines.py --dataset_version V6

```

