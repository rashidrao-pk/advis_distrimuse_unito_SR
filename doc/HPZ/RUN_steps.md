# Run ADVIS on the HP workstation

> HPZ documentation: [First-time setup](SETUP.md) · **Daily run** ·
> [ROS monitoring and recovery](RM_ROS.md) · [System services](SERVICES.md)

Use this checklist after the one-time installation in [SETUP.md](SETUP.md).
Unless stated otherwise, run each command on the HP Ubuntu workstation.

## 1. Connect and enter the repository

From your local machine:

```bash
ssh HPZ3_at_SR
```

On HP:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
```

You do not need to activate Conda. All project commands below use the Pixi
environment belonging to this repository.

When code or `pixi.lock` has changed, update before starting the runtime:

```bash
git pull
pixi install --locked
```

Rebuild the Cython extension after changing Python, NumPy, Cython, the lockfile,
or either TAAS `.pyx` file:

```bash
pixi run python scripts/setup_taas_cython.py build_ext --inplace
```

## 2. Choose manual mode or system-service mode

Use only one mode at a time:

- **Manual mode** is suitable for testing and development; continue below.
- **Service mode** is suitable for unattended operation; follow
  [SERVICES.md](SERVICES.md).

Check whether services are already running before launching duplicate manual
processes:

```bash
systemctl is-active vlan_setup.service
systemctl is-active inference.service
```

If `inference.service` is active and you want manual mode:

```bash
sudo systemctl stop inference.service
```

## 3. Prepare the SEDS VLAN and ROS network

```bash
cd /home/unito/dm/distrimuse-seds

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/unito/dm/distrimuse-seds/cyclonedds.xml
export ROS_DOMAIN_ID=1

source setup_ros.sh vlans.conf unito/dm kilted
./vlan_manager.sh vlans.conf
```

Return to ADVIS and restart ROS discovery:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
pixi run ros2 daemon stop
pixi run ros2 daemon start
```

Confirm that Pixi is activating the expected HP values:

```bash
pixi run python -c \
  "import os; print('RMW=', os.environ.get('RMW_IMPLEMENTATION')); print('DDS=', os.environ.get('CYCLONEDDS_URI')); print('DOMAIN=', os.environ.get('ROS_DOMAIN_ID'))"
```

## 4. Run the daily preflight checks

Verify ROS, OpenCV, CUDA, and the Cython TAAS extension:

```bash
pixi run python -c \
  "import rclpy, rosbag2_py, cv2; print('ROS/OpenCV: OK')"

pixi run python -c \
  "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"

PYTHONPATH=scripts pixi run python -c \
  "import tass_cython_distance; print('TAAS:', tass_cython_distance.__file__)"
```

Verify that checkpoints and thresholds are present:

```bash
pixi run python scripts/check_model_checkpoints.py \
  --config configs/cf_dataset_hp.yaml

for area in PLeft PRight RoboArm ConvBelt; do
  echo "===== $area ====="
  find "results/V6/thresholds/$area" -maxdepth 1 -type f \
    -name '*off3_sig1.5_q0.99*.json' | sort
done
```

## 5. Verify the camera input

```bash
pixi run ros2 topic list -t
pixi run ros2 topic type /camera/back_view/image_raw
pixi run ros2 topic info /camera/back_view/image_raw --verbose
pixi run ros2 topic hz /camera/back_view/image_raw
```

Stop `ros2 topic hz` with `Ctrl-C` after confirming a stable rate. If the topic
is absent or its rate is zero, use [RM_ROS.md](RM_ROS.md) before starting live
inference.

## 6. Optional offline smoke test

An offline test verifies models, thresholds, masks, and TAAS independently of
the live camera network.

### From a saved video

```bash
pixi run python scripts/infer_offline.py \
  --config configs/cf_dataset_hp.yaml \
  --input_type video \
  --scenario 13_1 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --threshold_calibration_mode auto \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --max_frames 200 \
  --skip-first 100 \
  --scores-only
```

### Directly from a rosbag

```bash
pixi run python scripts/infer_offline.py \
  --config configs/cf_dataset_hp.yaml \
  --input_type rosbag \
  --scenario 8_0 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --threshold_calibration_mode auto \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --max_frames 200 \
  --skip-first 100 \
  --scores-only
```

See [SETUP.md](SETUP.md#7-offline-inference-from-a-saved-video) for video
path rules and [SETUP.md](SETUP.md#8-offline-inference-directly-from-a-rosbag)
for MCAP diagnostics.

## 7. Start the live stack manually

Use separate terminals for the router, inference, and dashboard.

### Terminal A: Zenoh router

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
zenohd -c zenoh_dashboard/zenoh.json5
```

Leave this process running.

### Terminal B: live inference

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR

pixi run python scripts/inference_live.py \
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
  --log_every_n 1 \
  --profile_timing \
  --publish_zenoh \
  --debug_mode \
  --zenoh_endpoint tcp/127.0.0.1:7447 \
  --publish_rulex \
  --rulex_topic /rulex/data
```

The startup log must list one model and one threshold for every requested
safety area. It also reports whether each threshold was calibrated using
`test`, `val`, or a legacy file.

### Terminal C: dashboard

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR

python zenoh_dashboard/dashboard_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --camera-topic /camera/back_view/image_raw \
  --camera-message-type compressed \
  --camera-timeout 2 \
  --inference-timeout 3
```

## 8. Verify live outputs

Use a fourth terminal:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR

pixi run ros2 topic hz /camera/back_view/image_raw
pixi run ros2 topic hz /advis/detections
pixi run ros2 topic hz /rulex/data
```

Inspect one message at a time:

```bash
pixi run ros2 topic echo /advis/detections --once
pixi run ros2 topic echo /rulex/data --once
pixi run ros2 topic echo /rulex/data --field area_scores --once
```

More message, publisher, subscriber, and recovery checks are documented in
[RM_ROS.md](RM_ROS.md).

## 9. Stop a manual session

Stop the dashboard and inference with `Ctrl-C`, then stop the Zenoh router.
Confirm no duplicate manual processes remain:

```bash
pgrep -af 'inference_live.py|dashboard_viewer.py|zenohd'
```

If the deployment normally runs as services, restore them using
[SERVICES.md](SERVICES.md):

```bash
sudo systemctl enable --now vlan_setup.service
sudo systemctl enable --now inference.service
```
