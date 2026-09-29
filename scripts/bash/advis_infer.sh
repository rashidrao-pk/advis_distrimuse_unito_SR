#!/bin/bash
set -e

export HOME=/home/unito
export PATH="/home/unito/.pixi/bin:$PATH"

export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unito/dm/distrimuse-seds/cyclonedds.xml

# Load DistriMuSe ROS2 custom message definitions
source /home/unito/advis/distrimuse-ros2-api/install/setup.bash

cd /home/unito/advis/advis_distrimuse_unito_SR

echo "Starting ADVIS V6 live inference..."

exec pixi run python scripts/inference_live.py \
  --config configs/cf_dataset_hp.yaml \
  --camera_topic /camera/back_view/image_raw \
  --message_type auto \
  --safety_areas ALL \
  --threshold_calibration_mode auto \
  --threshold_strategy percentile \
  --threshold_amplification 1.1 1.1 1.1 1.2 \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --taas_backend auto \
  --taas_variant canonical \
  --rolling mean \
  --rolling_window 5 \
  --detections_topic /advis/detections \
  --log_every_n 10 \
  --profile_timing \
  --publish_zenoh \
  --debug_mode \
  --zenoh_endpoint tcp/127.0.0.1:7447 \
  --publish_rulex \
  --rulex_topic /rulex/data
