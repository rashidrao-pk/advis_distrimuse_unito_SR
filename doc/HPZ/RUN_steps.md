

1. ## RUN SAFROON 
```bash
cd ~/dm/distrimuse-seds/
source setup_ros.sh vlans.conf unito/dm kilted
./vlan_manager.sh vlans.conf

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