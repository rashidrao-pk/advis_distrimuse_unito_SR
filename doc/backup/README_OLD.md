# DistriMuSe UC3 – Real-Time Anomaly Detection for Human–Robot Safety

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?color=00E5C3&lines=Real-Time+Anomaly+Detection+for+Safe+Human-Robot+Interaction;ROS2+%7C+VAE-GAN+%7C+Industrial+Safety+Monitoring;Safety-Area+Inference+%7C+Thresholding+%7C+Alert+Publishing;University+of+Torino+%7C+DistriMuSe+Project&center=true&width=900&height=45">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-DistriMuSe-0A192F?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Use%20Case-UC3-00E5C3?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-ROS2-1f6feb?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-VAE--GAN-7A3EFF?style=for-the-badge" />
</p>

<p align="center">
  <img src="doc/header.png" width="100%">
</p>

## Overview

This repository contains the anomaly detection pipeline developed for **Use Case 3 (Safe Interaction with Robots)** within the DistriMuSe project.

The system uses **VAE/VAE-GAN models** to monitor predefined safety areas in collaborative robotics environments and detect unexpected conditions in real time.

### Features

- Safety-area-specific anomaly detection
- VAE / VAE-GAN training
- Automatic threshold calibration
- ROS2 integration
- Rulex-compatible message publishing
- Zenoh dashboard visualization
- Distributed deployment support
  w

---

## Pipeline

```text
Camera Stream
      ↓
Safety Area Extraction
      ↓
VAE-GAN Reconstruction
      ↓
Anomaly Score Computation
      ↓
Threshold Comparison
      ↓
ROS2 / Rulex Alert
```

---

## Safety Areas

| Area     | Description          |
| -------- | -------------------- |
| RoboArm  | Robot operating zone |
| ConvBelt | Conveyor belt zone   |
| PLeft    | Left pallet area     |
| PRight   | Right pallet area    |

---

## Repository Structure

```text
advis_distrimuse_unito_SR/
│
├── scripts/
│   ├── train.py
│   ├── calibrate_threshold.py
│   ├── infer_ros_live_zenoh.py
│   ├── preprocess_saved_frames.py
│   └── results/
│
├── pixi.toml
└── README.md
```

---

# Environment Setup

## Clone Repository

```bash
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito_SR.git

cd advis_distrimuse_unito_SR
```

## Install Environment

```bash
pixi install
```

## Verify Installation

```bash
pixi run python -c "import torch; print(torch.cuda.is_available())"

pixi run python -c "import cv2; print('OpenCV OK')"

pixi run python -c "import rclpy; print('ROS2 OK')"
```

---

# 1. Run in Distributed Environment

## Connect to DistriMuSe Infrastructure

```bash
ssh -X unito@distrimuse.idrago.org -p10022
```

---

## Configure VLAN

```bash
cd dm/distrimuse-seds/

source setup_ros.sh vlans.conf unito/dm kilted

./vlan_manager.sh vlans.conf
```

---

## Verify Camera Stream

```bash
export ROS_DOMAIN_ID=1

ros2 topic list

ros2 topic hz /camera/back_view/image_raw
```

## x`

## Verify Rulex Messages

```bash
source ~/advis/distrimuse-ros2-api/install/setup.bash

ros2 topic echo /rulex/data
```

---

## Run Production Inference

```bash
cd ~/advis/advis_distrimuse_unito_SR

pixi run python scripts/infer_ros_live_zenoh.py \
  --camera_topic /camera/back_view/image_raw \
  --safety_area PRight PLeft RoboArm \
  --area_names PRight PLeft RoboArm \
  --static_mask_paths \
    /home/unito/advis/DS/SR/v4/masks/Mask_Generation_v4_PRight.png \
    /home/unito/advis/DS/SR/v4/masks/Mask_Generation_v4_PLeft.png \
    /home/unito/advis/DS/SR/v4/masks/Mask_Generation_v4_RoboArm.png \
  --threshold_dir /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/thresholds_v4 \
  --checkpoints /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/models_v4 \
  --latent_dims 64 \
  --frame_stride 1 \
  --publish_rulex
```

---

## Monitor Detection Output

```bash
source ~/advis/distrimuse-ros2-api/install/setup.bash

ros2 topic echo /rulex/data
```

---

# 2. Run on Local Machine

## Prepare Dataset

### Replay ROS Bag

```bash
pixi run replay \
  /path/to/recording \
  --no-display
```

### Save Frames

```bash
pixi run python scripts/pixi/pixi_saveframes.py \
  --ros-args \
  -p save_dir:=/path/to/output \
  -p topics:="['/camera/back_view/image_raw']"
```

### Generate Safety Area Crops

```bash
pixi run python scripts/scripts_extra/preprocess_saved_frames.py \
  --input_dir INPUT_IMAGES \
  --save_dir OUTPUT_DATASET \
  --area_names PRight \
  --static_mask_paths \
    "/path/to/Mask_Generation_v4_PRight.png" \
  --save_every_n 1 \
  --image_format png \
  --keep_aspect True \
  --class_label normal
```

---

## Train Model

<summary> 

### PRight

```bash
python scripts/train.py \
  --safety_area PRight \
  --dataset_source SR \
  --dataset_version v4 \
  --dataset_cam_type back_view \
  --epochs 200 \
  --batch_size 16 \
  --latent_dims 64 \
  --augmentation_type custom
```

### PLeft

```bash
python scripts/train.py \
  --safety_area PLeft \
  --dataset_source SR \
  --dataset_version v4 \
  --dataset_cam_type back_view \
  --epochs 200 \
  --batch_size 16 \
  --latent_dims 64 \
  --augmentation_type custom
```

### RoboArm

```bash
python scripts/train.py \
  --safety_area RoboArm \
  --dataset_source SR \
  --dataset_version v4 \
  --dataset_cam_type back_view \
  --epochs 200 \
  --batch_size 16 \
  --latent_dims 64 \
  --augmentation_type custom
```

---------------------------------------------------------------------

## Calibrate Thresholds

### PRight

```bash


python scripts/calibrate_threshold.py \
  --mode val \
  --safety_area PRight \
  --dataset_version v6 \
  --dataset_type back_view
```

### PLeft

```bash
python scripts/calibrate_threshold.py \
  --mode val \
  --safety_area PLeft \
  --dataset_version v4 \
  --dataset_type back_view
```

### RoboArm

```bash
python scripts/calibrate_threshold.py \
  --mode val \
  --safety_area RoboArm \
  --dataset_version v4 \
  --dataset_type back_view
```

---

---------------------------------------------------------------------

## Run Local Inference

```bash
pixi run python scripts/infer_ros_live_zenoh.py \
  --camera_topic /camera/back_view/image_raw \
  --safety_area PRight PLeft RoboArm \
  --area_names PRight PLeft RoboArm \
  --static_mask_paths \
    /path/to/Mask_Generation_v4_PRight.png \
    /path/to/Mask_Generation_v4_PLeft.png \
    /path/to/Mask_Generation_v4_RoboArm.png \
  --threshold_dir scripts/results/thresholds_v4 \
  --checkpoints scripts/results/models_v4 \
  --latent_dims 64
```

---

## Threshold Calibration

Threshold calibration automatically determines:

- offset
- sigma
- quantile
- threshold

for each safety area.

Inference automatically loads these values from:

```text
scripts/results/thresholds_v4/<AREA>/threshold_<AREA>.json
```

No manual threshold tuning is required during deployment.

---

## Acknowledgements

This work was developed at the University of Torino within the DistriMuSe Project.

https://distrimuse.eu/

Special thanks to:

- University of Granada (ValeriaLab)
- Smart Robotics
- RuleX Innovation Labs

# 👥 Contributing

We welcome contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

<p align="center">
  <a href="https://github.com/rashidrao-pk/advis_distrimuse_unito_SR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=rashidrao-pk/advis_distrimuse_unito_SR" alt="Contributors to advis_distrimuse_unito_SR" />
  </a>
</p>

<p align="center">
  <b>Thank you to all our contributors!</b>
</p>
