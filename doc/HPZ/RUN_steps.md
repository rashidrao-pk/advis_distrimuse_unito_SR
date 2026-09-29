## 1. Connect Device remotely

```bash
ssh HPZ3_at_SR
```

## 2. RUN SEDS Vlan Setup

```bash
cd ~/dm/distrimuse-seds/
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_URI=file:///home/unito/dm/distrimuse-seds/cyclonedds.xml
ROS_DOMAIN_ID=1
source setup_ros.sh vlans.conf unito/dm kilted
./vlan_manager.sh vlans.conf

pixi run ros2 daemon stop
pixi run ros2 daemon start

```

### 1.3 Verify Installation

```bash
pixi run python -c "import rclpy; print('ROS OK')"
pixi run python -c "import cv2; print('CV OK')"
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

## Verify GPU being used:

```bash
pixi run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Inference:

```bash
conda actiavte dm_unito

#  RUN OFFLINE USING SAVED VIDEO
python scripts/infer_offline.py   --config configs/cf_dataset_hp.yaml   --input_type video   --scenario 13_1   --topic /camera/back_view/image_raw   --safety_areas ALL   --threshold_strategy percentile   --offset 1   --sigma 1.0   --quantile 0.99 --scores-only

#  RUN OFFLINE USING SAVED ROSBAG
pixi run python scripts/infer_offline.py \
  --config configs/cf_dataset_hp.yaml \
  --input_type rosbag \
  --scenario 8_0 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --max_frames 200 \
  --skip-first 100

```

## Run DASHBOARD VIEWER

```bash
python zenoh_dashboard/dashboard_viewer.py   --zenoh-endpoint tcp/127.0.0.1:7447   --camera-topic /camera/back_view/image_raw   --camera-message-type compressed   --camera-timeout 2   --inference-timeout 3
```
