

```bash
# Connect Server
ssh -Y unito@distrimuse

```



### 1.1 Retreive Codes 

```bash

cd ~/advis/
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito_SR
cd ~/advis/advis_distrimuse_unito_SR
```







### 1.2 Tranport Data

Ship Datasets to server

**Dataset v2**

```bash

# Copy Raw Videos 

scp D:\DS\SR\v2\camera1_20251210_142154_normal.mp4 unito@distrimuse:/home/unito/advis/DS/SR/v2

scp D:\DS\SR\v2\camera1_20251210_150045_box_fall.mp4 unito@distrimuse:/home/unito/advis/DS/SR/v2

scp D:\DS\SR\v2\camera1_20251210_151444_fallen_operator.mp4 unito@distrimuse:/home/unito/advis/DS/SR/v2

## Copy Processed Data
scp D:\DS\SR\v2\processed.zip unito@distrimuse:/home/unito/advis/DS/SR/v2
scp D:\DS\SR\v2\refined.zip unito@distrimuse:/home/unito/advis/DS/SR/v2
scp D:\DS\SR\v2\masks.zip unito@distrimuse:/home/unito/advis/DS/SR/v2

### TEST Data
scp D:\DS\SR\v2\camera1_20251210_150045_box_fall.zip unito@distrimuse:/home/unito/advis/DS/SR/v2
scp D:\DS\SR\v2\camera1_20251210_151444_fallen_operator.zip unito@distrimuse:/home/unito/advis/DS/SR/v2


cd /home/unito/advis/DS/SR/v2
unzip processed.zip -d /home/unito/advis/DS/SR/v2
unzip refined.zip -d /home/unito/advis/DS/SR/v2
unzip masks.zip -d /home/unito/advis/DS/SR/v2

unzip camera1_20251210_150045_box_fall.zip -d /home/unito/advis/DS/SR/v2
unzip camera1_20251210_151444_fallen_operator.zip -d /home/unito/advis/DS/SR/v2

rm camera1_20251210_150045_box_fall.zip
rm camera1_20251210_151444_fallen_operator.zip

```

#### ONLY ONCE
```bash
# INTALL DISPLAY Settings 
sudo apt install -y libxcb-cursor0
pixi run ros2 topic list
pixi run ros2 topic hz /camera/front_view/image_raw
pixi run ros2 topic echo /camera/front_view/image_raw --once
```







**Dataset V3**

```bash
scp D:/DS/SR/v3/recording_20260313_133316.zip unito@HPZ2miniDistriMuSe:
cd /home/unito/distrimuse
mkdir -p bags
unzip recording_20260313_133316.zip -d bags
find /home/unito/advis/bags/recording_20260313_133316 -maxdepth 2 -type f
```



**Check if OPen CV works**

```bash
# or

## check if the display for images is working
pixi run python -c "import cv2, numpy as np; img=np.zeros((300,300,3)); cv2.imshow('test', img); cv2.waitKey(0)"
```

```bash
cp ~/advis/advis_distrimuse_unito_SR/scripts/pixi_save_frames.py ~/advis/distrimuse-image-broadcaster
```



follow script or create to save_frames
```bash
nano pixi_save_frames.py
```



1️⃣  Custom save directory + one camera

```bash
pixi run python pixi_save_frames.py --ros-args -p save_dir:=/home/unito/advis/back_frames -p topics:="['/camera/back_view/image_raw']"
```

3️⃣ Multiple cameras 🔥

```bash
pixi run python pixi_save_frames.py --ros-args -p save_dir:=/home/unito/advis/all_frames -p topics:="['/camera/front_view/image_raw','/camera/back_view/image_raw','/camera/side_view/image_raw']"

```




```bash
# Verify saved frames
ls /home/unito/advis/back_frames | head 
```




### 1.2 Setup Image Streaming

a) Install Pixi first,
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
> Follow [this page](https://pixi.prefix.dev/latest/#installation) to setup and check if Pixi is working. 

b) use Smart Robotics Repo,

```bash
git clone https://github.com/smart-robotics/distrimuse-image-broadcaster
cd ~/advis/distrimuse-image-broadcaster
pixi install
cp config/config.yaml.example config/config.yaml
cp ~/advis/advis_distrimuse_unito_SR/scripts/pixi_save_frames.py ~/advis/distrimuse-image-broadcaster
```





### RUN IN ADVIS
Either 
 - 1. Initiate new Pixi env

```bash
pixi init
pixi add \
  python=3.12 \
  ros-jazzy-rclpy \
  ros-jazzy-sensor-msgs \
  ros-jazzy-std-msgs    \
  ros-jazzy-cv-bridge \
  opencv \
  numpy
```
Or
-2 Install from TOML file

```bash
# Step 2 — Add ROS2 + dependencies
cd ~/advis/advis_advis_distrimuse_unito_SR
# rm -rf .pixi
pixi install
# Step 1 — Create Pixi env inside ADVIS repo
```

### test ROS Installation

```bash
pixi run python -c "import rclpy; from sensor_msgs.msg import Image; print('ROS OK')"
```

** CHeck GPU Setup with PIXI

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```



### COPY DATA

```bash
# upload masks
scp -P 10022 "/Users/rashid/Mask Generation_PRight_MASK.png" \
unito@distrimuse.idrago.org:/home/unito/advis/DS/SR/v2/
```



```bash
# COUNT NUMBER OF FILES
ls /home/unito/advis/DS/SR/v2/train_processed/back_view/masked_input -l . | egrep -c '^-'
ls /home/unito/advis/DS/SR/v2/train_processed/back_view/ConvBelt -l . | egrep -c '^-'
ls /home/unito/advis/DS/SR/v2/train_processed/back_view/ConvBelt -l . | egrep -c '^-'
ls /home/unito/advis/DS/SR/v2/train_processed/back_view/ConvBelt -l . | egrep -c '^-'
```



## Pipeline Overview

```
Raw Video → Preprocessing → Training → Threshold Calibration → Inference
```


| Stage | Script | Description |
|---|---|---|
| Preprocess | — | Crop safety areas from 2540px-wide video, resize to 128×128 |
| Train | `scripts/train.py` | Train autoencoder per safety area |
| Compute Threshold | `scripts/compute_threshold.py` | Estimate reconstruction-error thresholds on validation data |
| Calibrate Threshold | `scripts/calibrate_threshold.py` | Tune thresholds using labelled or unlabelled data |
| Inference | `scripts/inference.py` | Run anomaly detection on preprocessed frames, raw frames, video, or live stream |

---






### 1.2 Clone GitHub Repo

```bash
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito_SR
conda activate dm_unito
pip install -r requirments.txt
```




### 1.3 Retreive Model Checkpoints (Demo 3.3)
Model Checkpoints are seperately provided and are available at following GitLab repo

-   [https://GitLab.di.unito.it/rashid/**_`dm_checkpoints_demo33`_**](https://gitlab.di.unito.it/rashid/dm_checkpoints_demo33)


#### Demo 3.3

Download Model checkpoints uploaded on following `GitLab` repo for `Real Palletizing` dataset for UC3 ( dataset provided by `Smart Robotics`, Netherlands for `DEMO-3.3` of UC3):
```bash
cd distrimuse_unito/scripts
git init
git pull https://gitlab.di.unito.it/rashid/dm_checkpoints_demo33    # FOR REAL ROBOT Palletizing - DEMO 3.3
cd ..
```