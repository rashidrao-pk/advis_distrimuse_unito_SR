#!/bin/bash

set -e
trap 'echo "Stopping all processes..."; kill 0' EXIT

export ROS_DOMAIN_ID=1
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# echo "Starting image broadcaster..."
# cd ~/advis/distrimuse-image-broadcaster || exit 1
# pixi run replay /home/unito/advis/bags/recording_20260313_133316/ --no-display --loop &
# sleep 5

cd /home/unito
source .bashrc
export PATH="/home/unito/.pixi/bin:$PATH"
export HOME=/home/unito

echo "Starting inference GUI..."
cd /home/unito/advis/advis_distrimuse_unito_SR

source /home/unito/advis/distrimuse-ros2-api/install/setup.bash

# echo pixi is `which pixi` and ROS_DOMAIN_ID is $ROS_DOMAIN_ID

pixi run python scripts/infer_ros_live_zenoh.py \
  --camera_topic /camera/back_view/image_raw \
  --safety_area PRight PLeft \
  --area_names PRight PLeft \
  --static_mask_paths \
    /home/unito/advis/DS/SR/v4/masks/Mask_Generation_v4_PRight.png \
    /home/unito/advis/DS/SR/v4/masks/Mask_Generation_v4_PLeft.png \
  --threshold_dir /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/thresholds_v4 \
  --checkpoints /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/models_v4 \
  --latent_dims 64 \
  --frame_stride 1 \
  --verbose_level 0 \
  --log_every_n 10 \
  --process_period 0.02 \
  --quantile 0.99 --offset 1 \
  --publish_rulex



