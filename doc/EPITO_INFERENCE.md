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

# Configured cropped training data
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type cropped \
  --safety_areas ALL \
  --max_frames 100


# Full-frame directory
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type frames \
  --input /path/to/frames \
  --safety_areas ALL

# MP4
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type video \
  --input /path/to/video.mp4 \
  --safety_areas PRight RoboArm \
  --frame_stride 5


# MCAP rosbag
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type rosbag \
  --input /beegfs/home/mrashid/datasets/AD/SR/V6/rosbags/Jul27_Scenario_13_0_2026-07-27_13-05-19/Jul27_Scenario_13_0_2026-07-27_13-05-19 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL

```

```bash
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type cropped \
  --safety_areas ALL \
  --max_frames 100

# --output_fps 10
# --timeline_history 500
# --timeline_seconds 4
# --output_video /custom/path/detections.mp4
# --timeline_png /custom/path/timeline.png


python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type rosbag \
  --input /beegfs/home/mrashid/datasets/AD/SR/V6/rosbags/Jul27_Scenario_13_1_2026-07-27_13-08-20 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --max_frames 1


python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type rosbag \
  --scenario 13_1 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --max_frames 200 \
  --skip-first 100

#  DOWNLOAD INFERENCE VIDEO ONLY
scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results/V6/offline_inference/rosbag_detections.mp4 ~/Downloads/

scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results/V6/offline_inference/rosbag_16_1_detections.mp4 ~/Downloads/

```


```bash
# DOWNLOAD ALL COMPUTED RESULTS
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

zip -r /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results_ADVIS_SR.zip results

scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results_ADVIS_SR.zip ~/Downloads/



