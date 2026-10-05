```bash
ssh HPZ3_at_SR
```

1. ## RUN SAFROON

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

### 1.2 Install Pixi Environment

```bash
cd ~/advis/advis_distrimuse_unito_SR
pixi install
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

##

```bash
python zenoh_dashboard/dashboard_viewer.py   --zenoh-endpoint tcp/127.0.0.1:7447   --camera-topic /camera/back_view/image_raw   --camera-message-type compressed   --camera-timeout 2   --inference-timeout 3
```

```bash
sudo systemctl stop inference.service

sudo systemctl is-active inference.service
pgrep -af 'inference_live.py|infer_ros_live'

```
