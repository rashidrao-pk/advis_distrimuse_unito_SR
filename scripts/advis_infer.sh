#!/bin/bash

set -e
trap 'echo "Stopping all processes..."; kill 0' EXIT

export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "Starting image broadcaster..."
cd ~/advis/distrimuse-image-broadcaster || exit 1
pixi run replay /home/unito/advis/bags/recording_20260313_133316/ --no-display --loop &

sleep 5

echo "Starting inference GUI..."
cd ~/advis/advis_distrimuse_unito_SR || exit 1
source /home/unito/advis/distrimuse-ros2-api/install/setup.bash

pixi run python scripts/infer_ros_live_GUI_v3.py \
  --camera_topic /camera/back_view/image_raw \
  --safety_area ALL \
  --area_names RoboArm ConvBelt PLeft PRight \
  --static_mask_paths \
    "/home/unito/advis/DS/SR/v3/masks/Mask Generation_RoboArm_MASK.png" \
    "/home/unito/advis/DS/SR/v3/masks/Mask Generation_ConvBelt_MASK.png" \
    "/home/unito/advis/DS/SR/v3/masks/Mask Generation_PLeft_MASK.png" \
    "/home/unito/advis/DS/SR/v3/masks/Mask Generation_PRight_MASK.png" \
  --threshold_dir /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/thresholds \
  --checkpoints /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/models_v2 \
  --latent_dims 64 \
  --frame_stride 1 \
  --verbose_level 1 \
  --log_every_n 10 \
  --process_period 0.02 \
  --show_timeline \
  --show_model_input