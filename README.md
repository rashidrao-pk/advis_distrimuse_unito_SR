# DistriMuSe-UC3

> **University of Torino** — Distributed Multi-Sensor Systems for Human Safety and Health

---

## Use Case 3 · Safe Interaction with Robots

An anomaly detection pipeline based on variational autoencoder models that monitors industrial safety areas in real time. The system processes video input, trains per-area models, calibrates detection thresholds, and runs inference on live or recorded footage.


## 1. Setup

Connect to remote device
```bash
ssh -X unito@distrimuse
```

### 1.1 Repo 

```bash
cd ~/advis/
cd ~/advis/advis_distrimuse_unito_SR
```


### Verify Models


---
Verify ROS-bag:
```bash
cd ~/advis/distrimuse-image-broadcaster
pixi run ros2 bag info /home/unito/advis/bags/recording_20260313_133316
```

Open Terminal and broadcast Frames


Now Play the video

```bash
pixi run replay /home/unito/advis/bags/recording_20260313_133316/
pixi run replay /home/unito/advis/bags/recording_20260313_133316/ --no-display

```

### 1.3 Replay or Save frames



and run following command to `Save` frames


```bash
cd ~/advis/distrimuse-image-broadcaster

pixi run python scripts/pixi/pixi_flow.py   \
    --ros-args   -p save_dir:=/home/unito/advis/DS/SR/v3/train_processed/back_view \
    -p camera_topic:=/camera/back_view/image_raw    \
    -p area_names:="['ConvBelt','PLeft','PRight','RoboArm']"    \
    -p static_mask_paths:="['/home/unito/advis/DS/SR/v2/masks/Mask Generation_ConvBelt_MASK.png','/home/unito/advis/DS/SR/v2/masks/Mask Generation_PLeft_MASK.png','/home/unito/advis/DS/SR/v2/masks/Mask Generation_PRight_MASK.png','/home/unito/advis/DS/SR/v2/masks/Mask Generation_RoboArm_MASK.png']" \
    -p save_every_n:=5  \
    -p image_format:=png    \
    -p keep_aspect:=true    \
    -p save_masked_full:=false  \
    -p save_masked_input:=false
```


### TRAIN USING NEW DATA THROGH ROS


```bash
python scripts/train.py \
  --safety_area PLeft \
  --dataset_source SR \
  --dataset_version v3 \
  --dataset_cam_type back_view \
  --epochs 200 \
  --batch_size 16 \
  --latent_dims 64 \
  --augmentation_type custom
```




### RUN IN ADVIS

**1. Install Pixi env**
```bash
pixi install
```

**2. test ROS Installation**

```bash
pixi run python -c "import rclpy; from sensor_msgs.msg import Image; print('ROS OK')"
```




Saved folders should be like this

```text
train_processed/
/home/unito/advis/DS/SR/v2/back_camera/
├── masked_input/
│   ├── masked_input_2026....png
│   ├── masked_input_2026....png
│   └── ...
├── ConvBelt/
├── PLeft/
├── PRight/
└── RoboArm/
```


```bash
cd ~/advis/distrimuse-image-broadcaster

pixi run python scripts/pixi/pixi_saveframes.py \
  --ros-args \
  -p save_dir:=/home/unito/advis/DS/SR/v3/back_view/train \
  -p camera_topic:=/camera/back_view/image_raw \
  -p area_names:="['ConvBelt','PLeft','PRight','RoboArm']" \
  -p static_mask_paths:="['/home/unito/advis/DS/SR/v2/masks/Mask Generation_ConvBelt_MASK.png','/home/unito/advis/DS/SR/v2/masks/Mask Generation_PLeft_MASK.png','/home/unito/advis/DS/SR/v2/masks/Mask Generation_PRight_MASK.png','/home/unito/advis/DS/SR/v2/masks/Mask Generation_RoboArm_MASK.png']" \
  -p save_every_n:=1 \
  -p image_format:=png \
  -p keep_aspect:=true \
  -p save_masked_full:=false \
  -p save_masked_input:=true \
  -p masked_input_subdir:=masked_input \
  -p masked_input_blur_ksize:=31 \
  -p masked_input_dim_factor:=0.35 \
  -p masked_input_outline_thickness:=6

```


## TRAIN FROM ROS Saved Data
```bash

python scripts/train.py \
  --safety_area PLeft \
  --dataset_source SR \
  --dataset_version v3 \
  --dataset_cam_type back_view \
  --epochs 200 \
  --batch_size 16 \
  --latent_dims 64 \
  --augmentation_type custom


```


TRAIN FOR ALL 


```bash
python scripts/train.py \
  --safety_area ALL \
  --dataset_source SR \
  --dataset_version v3 \
  --dataset_cam_type back_view \
  --epochs 20 \
  --batch_size 16 \
  --latent_dims 64 \
  --augmentation_type custom
```


```bash
# NEW MODEL
python scripts/calibrate_threshold.py --mode val --safety_area PLeft --dataset_version v3 --dataset_type back_view --checkpoints scripts/results

python scripts/calibrate_threshold.py --mode val --safety_area ALL --dataset_version v3 --dataset_type back_view --checkpoints scripts/results

```


## CHECK GPU BEFORE INF

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

```bash
pixi run python scripts/infer_ros_live.py
```

<!-- /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/models -->
<!-- /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/threshold/RoboArm/threshold_RoboArm.json
<!-- /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/thresholds/RoboArm/threshold_RoboArm.json -->


```bash
pixi run python scripts/infer_ros_live.py \
  --camera_topic /camera/back_view/image_raw \
  --safety_area ALL \
  --static_mask_paths \
    /home/unito/advis/DS/SR/v3/masks/Mask\ Generation_PLeft_MASK.png \
    /home/unito/advis/DS/SR/v3/masks/Mask\ Generation_PRight_MASK.png \
    /home/unito/advis/DS/SR/v3/masks/Mask\ Generation_ConvBelt_MASK.png \
    /home/unito/advis/DS/SR/v3/masks/Mask\ Generation_RoboArm_MASK.png \
  --threshold_dir /home/unito/advis/advis_distrimuse_unito_SR/scripts/results/threshold \
  --checkpoints ../advis_distrimuse_unito_SR/scripts/results/models \
  --latent_dims 64 \
  --frame_stride 5
```

**Verify Inference Setup**

```bash
pixi run python -c "import torch; print(torch.__version__)"
pixi run python -c "import rclpy; print('ROS OK')"
pixi run python -c "import cv2; print('CV OK')"
```

```bash
python scripts/inference.py \
    --data_source raw \
    --input_dir /advis/frames \
    --safety_area PLeft \
    --threshold_dir scripts/results/threshold \
    --save_figures
```

```bash
# COUNT NUMBER OF FILES
ls /home/unito/advis/DS/SR/v3/train/back_view/masked_input -l . | egrep -c '^-'
ls /home/unito/advis/DS/SR/v3/train/back_view/ConvBelt -l . | egrep -c '^-'
ls /home/unito/advis/DS/SR/v3/train/back_view/ConvBelt -l . | egrep -c '^-'
ls /home/unito/advis/DS/SR/v3/train/back_view/ConvBelt -l . | egrep -c '^-'
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

## Safety Areas

| ID | Description |
|---|---|
| `RoboArm` | Robot arm zone |
| `ConvBelt` | Conveyor belt zone |
| `PLeft` | Personnel zone — left |
| `PRight` | Personnel zone — right |
| `ALL` | Run all areas sequentially |

---

## ⚙️ 1. Setup
### 1.1 Create environment


```bash
conda create -n dm_unito python==3.9.18
conda activate dm_unito
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

### 1.2 Clone GitHub Repo

```bash
conda activate dm_unito
```



---

## 2. Train

Train an `VAE-GAN` model on one (`PLeft`, `PRight`, `RoboArm`, `ConvBelt`) or all safety areas.

```bash
# Single area (default settings)
python scripts/train.py --safety_area RoboArm
```

```bash
# All areas sequentially
python scripts/train.py --safety_area ALL
```

```bash
# All areas with custom settings
python scripts/train.py --safety_area ALL --epochs 200 --batch_size 64
```

**Full argument reference**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--safety_area` | `str` | `RoboArm` | Area to train (`RoboArm`, `ConvBelt`, `PLeft`, `PRight`, `ALL`) |
| `--epochs` | `int` | `1000` | Number of training epochs |
| `--batch_size` | `int` | `64` | Batch size (set automatically based on device if omitted) |
| `--data_split` | `int` | `80` | Train/validation split percentage (remainder used for validation) |
| `--augmentation_level` | `int` | `0` | `0` = none, `1` = custom augmentation |
| `--verbose_level` | `int` | `0` | `0` = silent, `1` = standard, `2` = detailed |
| `--save_figures` | flag | off | Save reconstruction plots, learning curves, and latent space visualisations |

### 2.1 Train with allow/ignore Intermediate Results

This will allow to train VAE-GAN models to allow/ignore intermediate results/figures to store.


```bash
# Fast mode — saves curves and checkpoints only
python scripts/train.py --safety_area RoboArm
```

```bash
# Full mode — also saves reconstruction figures
python scripts/train.py --safety_area RoboArm --save_figures
```

---

## 3. Compute/Calibrate Threshold

### 3.1 Compute Threshold with Validation-set (Subset of Train-set)
- Estimate per-area anomaly `thresholds` from `reconstruction errors` on the `validation` set (ratio used as `80/20`).

```bash
# Validation mode — no labels required
python scripts/calibrate_threshold.py --mode val --safety_area ALL --dataset_version v3 --dataset_type back_view

python scripts/calibrate_threshold.py --mode val --safety_area ALL --dataset_version v3 --dataset_type back_view
```

or 
### 3.2 Compute Threshold with Test-set
- Calibrate `thresholds` using labelled test set (containing both `normal` and `anomalous` data).

```bash
# Test mode — labelled CSV required
python scripts/calibrate_threshold.py --mode test --safety_area ALL \
    --gt_csv scripts/advis/annotations/anom_metadata_unexpected_person.csv
```

```bash
# Test mode — tune monitoring metric and search grid
python scripts/calibrate_threshold.py --mode test --safety_area RoboArm --gt_csv scripts/data/annotations/anom_metadata_unexpected_person.csv --monitor_score recall --offset_ls 1,2,3 --sigma_ls 0.0,0.5,1.0,1.5
```

---

## 4. Inference

Run anomaly detection from multiple `input sources` inluding following input sources;


### 4.1 Data source options

| `--data_source` | Input | Notes |
|---|---|---|
| `preprocessed` | Pre-cropped image folder | Fastest; requires prior preprocessing |
| `raw` | Raw frame folder | Crops and resizes on the fly |
| `video` | `.avi` / video file | Supports `--max_frames` limit |
| `ipcam` | RTSP stream URL | For live camera feeds |


### 4.2 Scripts:
```bash
# Pre-cropped frames, evaluate against annotations
python scripts/inference.py --data_source preprocessed --input_dir /home/unito/advis/DS/ValeriaLab/V6/fronttop/test_processed/unexpected_person --safety_area RoboArm --gt_csv scripts/data/annotations.csv
```

```bash
# Raw video frames, all areas, save output figures
python scripts/inference.py \
    --data_source raw \
    --input_dir /advis/frames \
    --safety_area PLeft \
    --save_figures
```

```bash
# Video file, process first 500 frames
python scripts/inference.py \
    --data_source video \
    --input_video /advis/test.avi \
    --max_frames 500
```

```bash
# Live IP camera stream, all areas
python scripts/inference.py \
    --data_source ipcam \
    --camera_url rtsp://192.168.1.10/stream \
    --safety_area ALL \
    --save_figures
```



---

## Repository Structure

```
distrimuse_unito/
├── scripts/
│   ├── train.py
│   ├── compute_threshold.py
│   ├── calibrate_threshold.py
│   ├── inference.py
│   └── results/                # Figures and threshold files
│       └── models/             # Saved model weights
│       └── threshold/          # Saved model weights
│       └── training/           # Saved Learning Curves
│   └── data/
│       └── annotations/

└── README.md
```

---

## Acknowledgements

Developed at the **University of Torino** as part of the [**_DistriMuse_**](https://distrimuse.eu/) project on distributed multi-sensor systems for human safety and health.
