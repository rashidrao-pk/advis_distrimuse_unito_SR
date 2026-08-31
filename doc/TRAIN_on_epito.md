# Training Models on Epito Machine

```bash
ssh epito-mercurio

tmux new -s AD_SR

## GET PREVIOUS Session
tmux ls
ps -ef | grep tmux
echo $TMUX

#  ATTACH with previous
tmux attach -t shapbpt

tmux list-panes -t shapbpt -F '#{pane_pid} #{pane_current_command} #{pane_current_path}'

srun -p epito --gres=gpu:a100:1 --pty bash
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/

sinfo --format="%P %G %C"

srun -p epito --gres=gpu:a100:1 --pty bash
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

squeue -u mrashid

squeue -u mrashid -o "%.18i %.12P %.20j %.8T %.10M %.10l %.6D %R"

sattach <JOB_ID>.0


srun --jobid=<JOB_ID> --pty bash

scancel 92873

sacct -j 92623 \
  --format=JobID,JobName%25,State,Elapsed,ExitCode,MaxRSS,NodeList

sinfo -N -p mirri,gracehopper,cascadelake,epito \
  -o "%.18N %.18P %.10T %.16G %.20C"

squeue -p mirri,gracehopper,cascadelake,epito \
  -o "%.12i %.12u %.18P %.18j %.8T %.15N %.12b %.20R"
# -------------------------------------
python3 scripts/train.py \
  --config configs/cf_dataset_epito.yaml \
  --checkpoints results/models/V6/
  --safety_area PLeft \
  --epochs 2 \
  --batch_size 128 \
  --augmentation_type custom \
  --save_figures \
  --verbose_level 1


python3 scripts/train.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area PLeft \
  --save_figures \
  --estimate_time \
  --batch_size 128 \



# Open a second local terminal and SSH to the login node:
ssh epito-mercurio

# First find your running job and assigned node:
squeue -u mrashid \
  -o "%.12i %.12P %.20j %.8T %.15N %.12b %.10M"

# connect to that node:
ssh epito04

# Then monitor GPU usage continuously:
watch -n 1 nvidia-smi
```
