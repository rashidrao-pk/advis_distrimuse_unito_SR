```bash

sinfo -N -p mirri,gracehopper,cascadelake,epito \
  -o "%.18N %.18P %.10T %.16G %.20C"

squeue -p mirri,gracehopper,cascadelake,epito \
  -o "%.12i %.12u %.18P %.18j %.8T %.15N %.12b %.20R"

ssh epito-mercurio
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

# /beegfs/home/mrashid/datasets/AD/SR/V6/videos/all_scenarios_after_7_2_back_view.mp4


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
  --scenario 8_0 \
  --threshold_strategy percentile \
  --safety_areas ALL


## RUN ALL INFERENCE
#-------------------------------------------------------------------
for sid in 8_0 8_1 8_2 8_3 8_4 9_0 10_0 10_1 11_0 11_1 12_0 12_1 13_0 13_1 14_1 14_1 15_0 16_0 16_1; do
  echo "=========================================================="
  echo "Processing scenario: $sid"

  python scripts/infer_offline.py \
    --config configs/cf_dataset_epito.yaml \
    --input_type rosbag \
    --scenario "$sid" \
    --threshold_strategy max \
    --safety_areas ALL

  echo "=========================================================="
done
#-------------------------------------------------------------------
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type rosbag \
  --scenario 13_1 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --max_frames 200 \
  --skip-first 100

# /beegfs/home/mrashid/datasets/AD/SR/V6/videos/s-8_16_c-back_view.mp4

python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type video \
  --scenario 8_16 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --offset 1 \
  --sigma 1.0 \
  --quantile 0.99 \
  --max_frames 200 \
  --skip-first 100 \
  --scores-only
```

## Ablation on Full Video

```bash
#  OFFSET, SIGMA, and QUANTILE variants
#!/usr/bin/env bash

set -euo pipefail

total=18
current=0

for ooff in 1 2 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))

      echo "================================================================"
      echo "Combination ${current}/${total}"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "================================================================"

      python scripts/infer_offline.py \
        --config configs/cf_dataset_epito.yaml \
        --input_type video \
        --scenario 8_16 \
        --topic /camera/back_view/image_raw \
        --safety_areas ALL \
        --threshold_strategy percentile \
        --threshold_percentile 99.0 \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" \
        --scores-only
    done

    echo "[COMPLETED] ✅ All quantiles for offset=${ooff}, sigma=${ss}"
  done

  echo "[COMPLETED] ✅ All sigma values for offset=${ooff}"
done

echo "[COMPLETED] ✅ All ${total} threshold combinations"

```

```bash
set -euo pipefail

total=6
current=0

for ooff in 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))

      echo "================================================================"
      echo "Combination ${current}/${total}"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "================================================================"

      python scripts/infer_offline.py \
        --config configs/cf_dataset_epito.yaml \
        --input_type video \
        --scenario 8_16 \
        --topic /camera/back_view/image_raw \
        --safety_areas ALL \
        --threshold_strategy percentile \
        --threshold_percentile 99.0 \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" \
        --scores-only
    done

    echo "[COMPLETED] ✅ All quantiles for offset=${ooff}, sigma=${ss}"
  done

  echo "[COMPLETED] ✅ All sigma values for offset=${ooff}"
done

echo "[COMPLETED] ✅ All ${total} threshold combinations"
```

```bash
chmod +x scripts/bash/run_inference.sh
./scripts/bash/run_inference.sh
```

```bash

#  DOWNLOAD INFERENCE VIDEO ONLY
scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results/V6/offline_inference/rosbag_detections.mp4 ~/Downloads/

scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results/V6/offline_inference/rosbag_16_1_detections.mp4 ~/Downloads/

```

````bash
# DOWNLOAD ALL COMPUTED RESULTS
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

zip -r /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results_ADVIS_SR.zip results

scp mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/repos/advis_distrimuse_unito_SR/results_ADVIS_SR.zip ~/Downloads/


> Check Anomalous Events

```bash

annotation="reports/safety_area_annotations/saved_annotation/scenario_9_0_back_view_annotations.csv"

awk -F',' '
  NR > 1 && tolower($8) ~ /anomalous/ {count++}
  END {print "Anomalous rows:", count + 0}
' "$annotation"
````

```bash
python scripts/compare_annotations_detection.py \
  --annotations reports/safety_area_annotations/saved_annotation/scenario_9_0_back_view_annotations.csv \
  --scores results/V6/offline_inference/rosbag_9_0_max_scores.csv

```

```bash
python scripts/compare_annotations_detection.py \
  --annotations reports/safety_area_annotations/saved_annotation/scenario_9_0_back_view_annotations.csv \
  --scores results/V6/offline_inference/rosbag_9_0_percentile_scores.csv

```

## RUN EVALUATION for ALL Scenarios

```bash
annotation_dir="reports/safety_area_annotations/saved_annotation"
score_dir="results/V6/offline_inference"

scenarios=(
  8_0 8_1 8_2 8_3 8_4
  9_0
  10_0 10_1
  11_0 11_1
  12_0 12_1
  13_0 13_1
  14_0 14_1
  15_0
  16_0 16_1
)

for sid in "${scenarios[@]}"; do
  annotation_file="${annotation_dir}/scenario_${sid}_back_view_annotations.csv"
  score_file="${score_dir}/rosbag_${sid}_max_scores.csv"

  echo "=========================================================="
  echo "Comparing scenario: ${sid}"

  if [[ ! -f "$annotation_file" ]]; then
    echo "Skipping: annotation file is missing"
    echo "  $annotation_file"
    continue
  fi

  if [[ ! -f "$score_file" ]]; then
    echo "Skipping: score file is missing"
    echo "  $score_file"
    continue
  fi

  python scripts/compare_annotations_detection.py \
    --annotations "$annotation_file" \
    --scores "$score_file"

  if [[ $? -ne 0 ]]; then
    echo "FAILED: scenario ${sid}"
  else
    echo "Completed: scenario ${sid}"
  fi
done
```

```bash
python scripts/compare_annotations_detection.py \
  --annotations /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  --scores /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/results/V6/offline_inference/video_8_16_percentile_off1_sig1.0_q0.99_scores.csv
```

```bash
set -euo pipefail

total=18
current=0

annotations="/Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv"

scores_dir="/Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/results/V6/offline_inference"

for ooff in 1,2,3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))

      scores="${scores_dir}/video_8_16_percentile_off${ooff}_sig${ss}_q${qq}_scores.csv"

      echo "================================================================"
      echo "Combination ${current}/${total}"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "Scores: ${scores}"
      echo "================================================================"

      if [[ ! -f "$scores" ]]; then
        echo "[ERROR] Score CSV does not exist: $scores"
        exit 1
      fi

      python scripts/compare_annotations_detection.py \
        --annotations "$annotations" \
        --scores "$scores"
    done

    echo "[COMPLETED] All quantiles for offset=${ooff}, sigma=${ss}"
  done

  echo "[COMPLETED] All sigma values for offset=${ooff}"
done

echo "[COMPLETED] All ${total} threshold combinations"
```

```bash
set -euo pipefail

total=6
current=0

annotations="/Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv"

scores_dir="/Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/results/V6/offline_inference"

for ooff in 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))

      scores="${scores_dir}/video_8_16_percentile_off${ooff}_sig${ss}_q${qq}_scores.csv"

      echo "================================================================"
      echo "Combination ${current}/${total}"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "Scores: ${scores}"
      echo "================================================================"

      if [[ ! -f "$scores" ]]; then
        echo "[ERROR] Score CSV does not exist: $scores"
        exit 1
      fi

      python scripts/compare_annotations_detection.py \
        --annotations "$annotations" \
        --scores "$scores"
    done

    echo "[COMPLETED] All quantiles for offset=${ooff}, sigma=${ss}"
  done

  echo "[COMPLETED] All sigma values for offset=${ooff}"
done

echo "[COMPLETED] All ${total} threshold combinations"

```

```bash
chmod +x scripts/bash/run_evaluation.sh
./scripts/bash/run_evaluation.sh
```

## Run on Winning without RollingMin

```bash
python scripts/compare_annotations_detection.py \
  --annotations /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  --scores results/V6/offline_inference/video_8_16_percentile_off3_sig1.5_q0.99_scores.csv

python scripts/compare_annotations_detection.py \
  --annotations /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  --scores results/V6/offline_inference/video_8_16_percentile_off3_sig1.5_q0.99_taas-minimization_scores.csv

```

New params added:

- `--add_score_name`
- `--add_fps_details`
- `--profile-timing`
- `--taas_backend` --> `cython` | `numpy`
- `--rolling` --> `mean` | `min` | `max` or `none`
- `--taas_backend` `numpy` | `cython`
- `--taas_variant` `canonical` | `minimization`

## Run on Winning with RollingMin

#### Rolling:

- `mean`: smooths short spikes and dips.
- `min`: requires all recent frames to remain high, giving the strongest false-alarm suppression but more detection delay.
- `max`: preserves any recent peak, improving short-anomaly sensitivity but keeping alarms active longer.
- `none`: original instantaneous behavior.

```bash
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type video \
  --scenario 8_16 \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --add_score_name \
  --add_fps_details \
  --rolling mean \
  --rolling_window 5 \
  --taas_backend cython \
  --add_score_name \
  --profile_timing
```

```bash
pixi add cython
pixi run python scripts/setup_taas_cython.py build_ext --inplace

pixi run python -c \
  "import sys; sys.path.insert(0, 'scripts'); import tass_cython_distance; print(tass_cython_distance.__file__)"

```

#### USING TAAS `MINIMIZATION`

```bash
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type video \
  --scenario 8_16 \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --threshold_percentile 99.0 \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --taas_variant minimization \
  --taas_backend cython \
  --rolling none \
  --add_score_name \
  --profile_timing
```

### Inference LIVE:

```bash
pixi run python scripts/inference_live.py \
  --config configs/cf_dataset_mac.yaml \
  --camera_topic /camera/back_view/image_raw \
  --message_type auto \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --offset 1 \
  --sigma 1.0 \
  --quantile 0.99 \
  --taas_backend auto \
  --taas_variant canonical \
  --rolling mean \
  --rolling_window 5 \
  --publish_rulex \
  --rulex_topic /rulex/data \
  --detections_topic /advis/detections \
  --log_every_n 1 \
  --profile_timing \
  --publish_zenoh \
  --debug_mode \
  --zenoh_endpoint tcp/127.0.0.1:7447

```

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR

zenohd -c zenoh_dashboard/zenoh.json5

### Verify
lsof -nP -iTCP:7447 -sTCP:LISTEN

```

## Start Zenoh Router:

```bash
zenohd -c zenoh_dashboard/zenoh.json5
```

```bash
python zenoh_dashboard/dashboard_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --zenoh-key advis/vis/dashboard/state
```

#### Scores Only:

```bash
python zenoh_dashboard/timeline_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --zenoh-key advis/vis/timeline/state
```

#### Debug Mode:

```bash
env -u DISPLAY QT_QPA_PLATFORM=cocoa \
  pixi run python zenoh_dashboard/debug_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --zenoh-key advis/vis/debug/state
```
