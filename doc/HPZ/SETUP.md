# ADVIS setup and inference on the HP workstation

> HPZ documentation: **First-time setup** · [Daily run](RUN_steps.md) ·
> [ROS monitoring and recovery](RM_ROS.md) · [System services](SERVICES.md)

This guide configures the HP Linux workstation for ADVIS training and
inference. Run repository commands from:

```text
/home/unito/advis/advis_distrimuse_unito_SR
```

The examples assume the V6 dataset is stored under:

```text
/home/unito/advis/DS/SR/v6
```

These locations agree with `configs/cf_dataset_hp.yaml`.

## 1. Update the repository and install the environment

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
git pull
pixi install
```

Always run project programs through `pixi run`. Do not rely on an activated
Conda environment or use bare `python`: it may select Python from another
repository and omit ROS packages such as `rclpy` and `rosbag2_py`.

Confirm which Python and ROS packages Pixi provides:

```bash
pixi run which python

pixi run python -c \
  "import sys; print(sys.executable)"

pixi run python -c \
  "import rclpy, rosbag2_py; print('ROS 2 Python dependencies: OK')"

pixi run python -c \
  "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The Python executable should be inside this repository's `.pixi/envs/`
directory, not the environment of `distrimuse-ros2-api` or another project.

### One Pixi manifest for Ubuntu and macOS

The committed `pixi.toml` and `pixi.lock` support both machines. Do not run
`pixi workspace platform add` after cloning: both platforms are already
declared.

Pixi selects the correct locked platform automatically:

- `linux-64`: PyTorch with CUDA 12.1, MKL, and the HP CycloneDDS file;
- `osx-arm64`: PyTorch without CUDA/MKL, its built-in MPS backend, and
  `configs/cyclonedds-macos.xml`.

The `.pixi` installation directory is local to each clone/machine and is not
shared through Git. Only `pixi.toml` and `pixi.lock` are shared, so installing
on the Mac does not replace the HP environment, and installing on HP does not
replace the Mac environment.

On either machine, use the same commands:

```bash
pixi install --locked
pixi run python your_script.py
```

Verify the HP CUDA environment:

```bash
pixi run python -c \
  "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Verify the Mac MPS environment:

```bash
pixi run python -c \
  "import torch; print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

The application already chooses devices in the order CUDA, MPS, then CPU.
`MPS built: True` confirms MPS support exists in PyTorch; `MPS available` also
requires an Apple Silicon GPU and a compatible macOS runtime.

Check the per-machine DDS configuration:

```bash
pixi run python -c \
  "import os; print(os.environ.get('CYCLONEDDS_URI')); print('ROS_DOMAIN_ID=', os.environ.get('ROS_DOMAIN_ID'))"
```

## 2. Build and test the Cython TAAS extension

The extension must be compiled with the same Python used for inference.

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR

pixi run python -c \
  "import Cython; print('Cython:', Cython.__version__)"

pixi run python scripts/setup_taas_cython.py build_ext --inplace
ls -lh scripts/tass_cython_distance*.so
```

Verify that inference can load it:

```bash
PYTHONPATH=scripts pixi run python -c \
  "import tass_cython_distance; print('Cython TAAS extension: OK')"
```

Run its focused tests when `pytest` is available in the environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pixi run python -m pytest tests/test_taas_cython.py -v
```

If Cython is unexpectedly missing, synchronize the committed environment and
rebuild. Do not install it with the `python` command from a separate Conda
environment:

```bash
pixi install --locked
pixi run python -c "import Cython; print(Cython.__version__)"
pixi run python scripts/setup_taas_cython.py build_ext --inplace
```

## 3. Prepare the V6 dataset

The important HP paths are:

```text
/home/unito/advis/DS/SR/v6/Jul27/train
/home/unito/advis/DS/SR/v6/Jul27/test
/home/unito/advis/DS/SR/v6/rosbags
/home/unito/advis/DS/SR/v6/videos
/home/unito/advis/DS/SR/v6/extracted_frames
/home/unito/advis/DS/SR/v6/masks
```

### Transfer the training dataset from macOS

Run these commands on the Mac:

```bash
tar -C /Users/rashid/data/DS/SR/v6/Jul27 \
  -czf /tmp/advis_v6_train.tar.gz train

scp /tmp/advis_v6_train.tar.gz \
  HPZ3_at_SR:/home/unito/advis/DS/SR/v6/Jul27/
```

Then run on the HP workstation:

```bash
mkdir -p /home/unito/advis/DS/SR/v6/Jul27

tar -C /home/unito/advis/DS/SR/v6/Jul27 \
  -xzf /home/unito/advis/DS/SR/v6/Jul27/advis_v6_train.tar.gz

ls -lah /home/unito/advis/DS/SR/v6/Jul27/train
du -sh /home/unito/advis/DS/SR/v6/Jul27/train
```

This extraction produces `Jul27/train`, not `Jul27/train/train`.

## 4. Download model checkpoints from Hugging Face

The repository is `rashidrao/ADVIS_SR_DISTRIMUSE`. Download the checkpoint
folder into a temporary directory and copy its contents into the path expected
by `cf_dataset_hp.yaml`:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR

pixi run hf --version
mkdir -p results/V6/train/models

pixi run hf download rashidrao/ADVIS_SR_DISTRIMUSE \
  --include "checkpoints/**" \
  --local-dir /tmp/advis_hf_checkpoints

cp -a /tmp/advis_hf_checkpoints/checkpoints/. results/V6/train/models/
rm -rf /tmp/advis_hf_checkpoints
```

Check the downloaded models:

```bash
find results/V6/train/models -maxdepth 1 -type f -name 'model_*.pt' | sort

pixi run python scripts/check_model_checkpoints.py \
  --config configs/cf_dataset_hp.yaml

pixi run python scripts/check_model_checkpoints.py \
  --config configs/cf_dataset_hp.yaml \
  --json
```

Run a small reconstruction test:

```bash
pixi run python scripts/test_model_inference.py \
  --config configs/cf_dataset_hp.yaml \
  --data_source training \
  --safety_area PLeft PRight RoboArm ConvBelt \
  --max_images 32
```

## 5. Download calibrated thresholds from Hugging Face

The Hugging Face repository contains a top-level `thresholds/` directory.
Download it directly into `results/V6` so the safety-area directory structure
is retained:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
mkdir -p results/V6

pixi run hf download rashidrao/ADVIS_SR_DISTRIMUSE \
  --include "thresholds/**" \
  --local-dir results/V6
```

Verify the result:

```bash
find results/V6/thresholds -maxdepth 2 \
  -type f -name 'threshold*.json' | sort
```

For example, verify the `offset=3`, `sigma=1.5`, `quantile=0.99`
configuration:

```bash
for area in PLeft PRight RoboArm ConvBelt; do
  echo "===== $area ====="
  find "results/V6/thresholds/$area" -maxdepth 1 -type f \
    -name '*off3_sig1.5_q0.99*.json' | sort
done
```

Threshold results are normally ignored by Git, so `git pull` alone does not
install them.

## 6. Understand threshold selection

Inference validates all of the following against the selected threshold JSON:

- calibration source: `test` or `val`;
- threshold strategy: `percentile`, `max`, `mean_std`, or `f1c`;
- TAAS variant: `canonical` or `minimization`;
- TAAS `offset`, `sigma`, and `quantile`.

Choose the calibration source with:

```text
--threshold_calibration_mode auto   # test, then val, then legacy
--threshold_calibration_mode test   # require a test threshold
--threshold_calibration_mode val    # require a validation threshold
```

`auto` may select different sources for different safety areas. The startup
log prints the exact threshold filename used for each area. A strict `test` or
`val` request fails instead of silently using the other source.

Example startup output:

```text
[threshold] PLeft: ... threshold_test_PLeft_....json
[threshold] RoboArm: ... threshold_val_RoboArm_....json
```

The default output filename also records the actual source, for example:

```text
rosbag_8_0_percentile_off3_sig1.5_q0.99_cal-test-val_scores.csv
```

## 7. Offline inference from a saved video

When `--scenario 13_1` is used, the script checks these paths beneath
`data.dataset_base`:

```text
extracted_frames/13_1/back_view/video/s-13_1_c-back_view.mp4
videos/s-13_1_c-back_view.mp4
```

Run scores-only inference:

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

If the video is outside the configured dataset layout, provide it directly and
omit `--scenario`:

```bash
pixi run python scripts/infer_offline.py \
  --config configs/cf_dataset_hp.yaml \
  --input_type video \
  --input /absolute/path/to/s-13_1_c-back_view.mp4 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --threshold_calibration_mode auto \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --scores-only
```

Search for a missing scenario video with:

```bash
find /home/unito/advis/DS/SR/v6 \
  -type f -iname '*13_1*back_view*.mp4'
```

Do not combine `--input` and `--scenario`.

## 8. Offline inference directly from a rosbag

Use the project Pixi environment because it supplies `rclpy`, `rosbag2_py`,
and the MCAP storage plugin:

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

Remove `--scores-only` when detection and timeline videos are also required.

### Diagnose rosbag or MCAP errors

First confirm the ROS reader is available:

```bash
pixi run python -c \
  "import rclpy, rosbag2_py; print('ROS bag reader: OK')"
```

Inspect a bag before inference:

```bash
bag_dir="/home/unito/advis/DS/SR/v6/rosbags/Jul27_Scenario_8_0_2026-07-27_11-55-34"

find "$bag_dir" -maxdepth 1 -type f -name '*.mcap' -exec ls -lh {} \;
pixi run ros2 bag info "$bag_dir"
```

If the generic MCAP reader reports `RecordLengthLimitExceeded`, check the file
header:

```bash
mcap_file=$(find "$bag_dir" -maxdepth 1 -type f -name '*.mcap' | head -n 1)
xxd -l 8 "$mcap_file"
```

A valid MCAP file begins with:

```text
89 4d 43 41 50 30 0d 0a
```

An invalid header usually means that the file is corrupt, incomplete, or is
not actually MCAP data. If the header is valid but bare `python` reports that
`rclpy` is missing, rerun with `pixi run python`.

## 9. Live ROS inference

Example live inference with rolling-score stabilization and Zenoh output:

```bash
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

Verify the input camera before starting inference:

```bash
pixi run ros2 topic type /camera/back_view/image_raw
pixi run ros2 topic hz /camera/back_view/image_raw
```

## 10. Zenoh dashboard

Start the Zenoh router in one terminal:

```bash
zenohd -c zenoh_dashboard/zenoh.json5
```

Start live inference in a second terminal, then start the dashboard viewer in a
third terminal:

```bash
pixi run python zenoh_dashboard/dashboard_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --camera-topic /camera/back_view/image_raw \
  --camera-message-type compressed \
  --camera-timeout 2 \
  --inference-timeout 3
```

If the viewer cannot connect, confirm that the router is listening:

```bash
ss -ltnp | grep 7447
```

## 11. Quick troubleshooting reference

### `ModuleNotFoundError: No module named 'rclpy'`

The wrong Python environment is active. Use:

```bash
pixi run python scripts/infer_offline.py ...
```

### `Threshold config not found ... Available: none`

Download thresholds, verify the requested offset/sigma/quantile, and check:

```bash
find results/V6/thresholds -maxdepth 2 -type f -name 'threshold*.json' | sort
```

### `Scenario video not found`

Copy/generate the expected video, or use `--input /absolute/video/path.mp4`
without `--scenario`.

### `RecordLengthLimitExceeded`

Run through Pixi first. If it persists, inspect the bag using
`pixi run ros2 bag info` and verify the MCAP magic bytes as described above.

### Cython backend is not built

```bash
pixi run python scripts/setup_taas_cython.py build_ext --inplace
```

Then confirm:

```bash
PYTHONPATH=scripts pixi run python -c \
  "import tass_cython_distance; print(tass_cython_distance.__file__)"
```

## Next step

After this one-time setup succeeds, use [RUN_steps.md](RUN_steps.md) for each
normal session. If ROS discovery or messages stop, follow
[RM_ROS.md](RM_ROS.md). For unattended startup, logs, or service control, use
[SERVICES.md](SERVICES.md).
