# DistriMuSe UC3: Real-Time Anomaly Detection for Human–Robot Safety

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
  <img src="doc/header.png" width="100%" alt="DistriMuSe UC3 anomaly-detection pipeline - UniTo, Muhammad Rashid Rao">
</p>

This repository contains the University of Torino anomaly-detection pipeline for
**DistriMuSe Use Case 3: Safe Interaction with Robots**. It trains one VAE-GAN
per monitored safety area, calibrates anomaly thresholds from validation data,
and performs real-time inference from a ROS 2 camera stream. Detection results
can be published to Rulex and visualized through Zenoh.

## System overview

```text
ROS 2 camera stream
        │
        ▼
Safety-area masks and crops
        │
        ▼
Area-specific VAE-GAN reconstruction
        │
        ▼
Anomaly score and calibrated threshold
        │
        ├──► Rulex ROS 2 message
        └──► Zenoh dashboard and timeline
```

The four supported safety areas are:

| Area | Monitored region |
| --- | --- |
| `PLeft` | Left pallet area |
| `PRight` | Right pallet area |
| `RoboArm` | Robot operating area |
| `ConvBelt` | Conveyor-belt area |

## Main entry points

| Script | Purpose |
| --- | --- |
| `scripts/train.py` | Train or resume an area-specific VAE-GAN |
| `scripts/calibrate_threshold.py` | Calculate validation or supervised test thresholds |
| `scripts/plot_validation_timelines.py` | Plot validation-score timelines and thresholds |
| `scripts/infer_offline.py` | Run inference on cropped data, frames, MP4, or MCAP rosbag input |
| `scripts/infer_ros_live_zenoh.py` | Run live ROS 2 inference and publish visualization data |
| `scripts/process_rosbags_to_dataset.py` | Build datasets from ROS bag recordings |
| `scripts/scripts_extra/preprocess_saved_frames.py` | Generate safety-area crops from saved frames |

## Installation

Clone the repository and create its Pixi environment:

```bash
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito_SR.git
cd advis_distrimuse_unito_SR
pixi install
```

Verify the principal dependencies:

```bash
pixi run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
pixi run python -c "import cv2; print('OpenCV:', cv2.__version__)"
pixi run python -c "import rclpy; print('ROS 2 Python: OK')"
```

The Pixi environment targets Linux, Python 3.12, PyTorch with CUDA 12.1, and
ROS 2 Kilted. Cluster users can instead activate the preconfigured environment
described in [Training on Epito](doc/TRAIN_on_epito.md).

## Configuration

Dataset and model locations are defined in YAML files under `configs/`. The
Epito configuration is [configs/cf_dataset_epito.yaml](configs/cf_dataset_epito.yaml):

```yaml
data:
  dataset_base: /path/to/dataset/V6
  masks: /path/to/dataset/V6/masks
  training: /path/to/dataset/V6/train
  testing: /path/to/dataset/V6/test

models:
  checkpoints: results/V6/models_V6
  latent_dims: 64
```

Training data must contain one ImageFolder-compatible directory per safety area.
For example:

```text
train/
├── PLeft/
│   └── normal/
├── PRight/
│   └── normal/
├── RoboArm/
│   └── normal/
└── ConvBelt/
    └── normal/
```

Paths supplied explicitly on the command line take precedence where the script
provides a corresponding override.

## Workflow

Run commands from the repository root.

### 1. Prepare the dataset

To extract safety-area crops from saved images:

```bash
pixi run python scripts/scripts_extra/preprocess_saved_frames.py \
  --input_dir /path/to/input/images \
  --save_dir /path/to/output/dataset \
  --area_names PRight \
  --static_mask_paths /path/to/PRight_mask.png \
  --save_every_n 1 \
  --image_format png \
  --keep_aspect True \
  --class_label normal
```

See [New dataset preparation](doc/Newdata.md) for the broader ingestion and
preprocessing workflow.

### 2. Train the models

Train one safety area with the Epito dataset configuration:

```bash
python scripts/train.py \
  --config configs/cf_dataset_epito.yaml \
  --dataset_version V6 \
  --safety_area PRight \
  --epochs 200 \
  --batch_size 128 \
  --augmentation_type custom \
  --save_figures
```

Use `--safety_area ALL` to train all four areas sequentially. Checkpoints use
the safety area and latent dimension in their names:

```text
results/V6/models_V6/model_PLeft_64.pt
results/V6/models_V6/model_PRight_64.pt
results/V6/models_V6/model_RoboArm_64.pt
results/V6/models_V6/model_ConvBelt_64.pt
```

Training resumes from a matching checkpoint when one is present. See
[Training on Epito](doc/TRAIN_on_epito.md) for Slurm, `tmux`, and GPU-monitoring
commands.

### 3. Calibrate thresholds

Calibrate one area from its normal validation split:

```bash
python scripts/calibrate_threshold.py \
  --config configs/cf_dataset_epito.yaml \
  --mode val \
  --safety_area PRight \
  --dataset_version V6
```

Use `--safety_area ALL` to process every area. Calibration reads the checkpoint
directory and latent dimension from the selected YAML configuration and loads
the matching `model_<area>_<latent_dims>.pt` file.

Results are written under `results/V6/thresholds/`:

```text
thresholds/
├── PRight/
│   ├── threshold_PRight.json
│   └── val_scores_PRight_off1_sig0.5_q1.0.csv
└── thresholds_summary_val_max.csv
```

For calibration strategies and supervised test mode, see
[Threshold calibration](doc/CALIBRATE_THRESHOLD.md).

### 4. Plot validation timelines

Generate a combined timeline for all safety areas:

```bash
python scripts/plot_validation_timelines.py --dataset_version V6
```

Add `--individual` to also create one PNG inside each area directory:

```bash
python scripts/plot_validation_timelines.py \
  --dataset_version V6 \
  --individual
```

The combined plot is saved as
`results/V6/thresholds/val_scores_timeline.png`.

### 5. Run offline inference

Use configured cropped training data (no masks required). Each output frame combines all selected safety areas in the Input, Unexpected Situations, and AI views:

```bash
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type cropped \
  --safety_areas ALL
```

For full frames, MP4 video, or an MCAP rosbag, select `frames`, `video`, or
`rosbag` and pass `--input`. Safety-area masks are discovered from the configured
`data.masks` directory; use repeated `--mask AREA=PATH` arguments to override them.

```bash
python scripts/infer_offline.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type video \
  --input /path/to/video.mp4 \
  --safety_areas PRight RoboArm \
  --frame_stride 5
```

Rosbag mode also accepts `--topic /camera/back_view/image_raw`. By default, the run writes a score CSV, a four-panel ADVIS dashboard MP4, and a final timeline PNG under `results/V6/offline_inference/`. The final timeline is also held at the end of the MP4; control its duration with `--timeline_seconds`.

### 6. Run live inference

The number and order of `--area_names` entries must match the mask paths:

```bash
pixi run python scripts/infer_ros_live_zenoh.py \
  --camera_topic /camera/back_view/image_raw \
  --safety_area PRight PLeft RoboArm ConvBelt \
  --area_names PRight PLeft RoboArm ConvBelt \
  --static_mask_paths \
    /path/to/PRight_mask.png \
    /path/to/PLeft_mask.png \
    /path/to/RoboArm_mask.png \
    /path/to/ConvBelt_mask.png \
  --threshold_dir results/V6/thresholds \
  --checkpoints results/V6/models_V6 \
  --latent_dims 64 \
  --frame_stride 1 \
  --publish_rulex
```

Confirm the camera and Rulex topics in another ROS 2-enabled terminal:

```bash
export ROS_DOMAIN_ID=1
ros2 topic hz /camera/back_view/image_raw
ros2 topic echo /rulex/data
```

See [Epito inference](doc/INFERENCE_Epito.md),
[local deployment](doc/deployment_LOCAL.md), and
[distributed deployment](doc/deployment_DISTRBUTED.md) for environment-specific
instructions.

## Outputs

The V6 workflow uses the following layout:

```text
results/V6/
├── models_V6/             # model_<area>_<latent_dims>.pt
├── training/              # loss curves and training artifacts
├── monitor/               # optional reconstruction figures
└── thresholds/            # score CSVs, threshold JSONs, and timelines
```

Generated datasets, checkpoints, and result files can be large. Keep deployment
paths in configuration files and avoid committing generated artifacts unless
they are intentionally part of a release.

## Troubleshooting

- **Checkpoint not found:** verify `models.checkpoints` and `models.latent_dims`
  in the selected YAML. The expected file is
  `model_<safety_area>_<latent_dims>.pt`.
- **Validation split not found:** run training first. It creates
  `split_4train_1val_<area>.json` inside the configured area training directory.
- **CUDA unavailable:** check the allocated GPU, CUDA-enabled PyTorch build, and
  `torch.cuda.is_available()`.
- **No camera frames:** verify `ROS_DOMAIN_ID`, CycloneDDS configuration, VLAN,
  and the camera topic with `ros2 topic hz`.
- **No Rulex messages:** source the DistriMuSe ROS 2 API workspace and pass
  `--publish_rulex` to live inference.

## Documentation

- [First-time setup](doc/README_FIRST_TIME.MD)
- [Training on Epito](doc/TRAIN_on_epito.md)
- [Threshold calibration](doc/CALIBRATE_THRESHOLD.md)
- [Inference on Epito](doc/INFERENCE_Epito.md)
- [Local deployment](doc/deployment_LOCAL.md)
- [Distributed deployment](doc/deployment_DISTRBUTED.md)
- [macOS setup](doc/README_Setup_mac.MD)

## Acknowledgements

This work was developed at the University of Torino within the
[DistriMuSe project](https://distrimuse.eu/), with contributions from project
partners including the University of Granada (ValeriaLab), Smart Robotics, and
Rulex Innovation Labs.


## License

See [LICENSE](LICENSE).

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