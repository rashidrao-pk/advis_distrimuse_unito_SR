# macOS (Apple Silicon) Setup

The image broadcaster was originally developed and tested on **Ubuntu Linux**. When running on **macOS (Apple Silicon)**, CycloneDDS may fail to initialize because the default Linux network interface configuration references interfaces that do not exist on macOS.

Typical error:

```text
rmw_create_node: failed to create domain
does not match an available interface
```

This can be resolved by using the provided macOS CycloneDDS configuration.

## 1. Copy the CycloneDDS configuration

Copy the macOS configuration file from the ADVIS repository:

```bash
cp \
~/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/configs/cyclonedds-macos.xml \
~/data/PhD/datacloud_data/repos/DistriMuSe/distrimuse-image-broadcaster/
```

Your broadcaster directory should now contain:

```
distrimuse-image-broadcaster/
├── cyclonedds-macos.xml
├── pixi.toml
├── src/
└── ...
```

---

## 2. Export the CycloneDDS configuration

Before running any ROS2 command, export:

```bash
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds-macos.xml
```

(Optional) For local rosbag replay only:

```bash
export ROS_LOCALHOST_ONLY=1
```

---

## 3. Verify the configuration

```bash
echo $CYCLONEDDS_URI
```

Expected output:

```text
file:///.../distrimuse-image-broadcaster/cyclonedds-macos.xml
```

---

## 4. Replay a rosbag

```bash
pixi run replay /path/to/your_recording.mcap
```

---

## 5. View the images

Open another terminal:

```bash
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds-macos.xml

pixi run view
```

or

```bash
pixi run view --topic /camera/back_view/image_raw
```

```bash
pixi run ros2 bag info /home/unito/advis/bags/recording_20260313_133316
```

```bash
env -u DISPLAY QT_QPA_PLATFORM=cocoa pixi run check-input \
  --camera_topic /camera/back_view/image_raw \
  --use_compressed \
  --safety_area PLeft PRight \
  --area_names PLeft \
  --static_mask_paths \
    /Users/rashid/data/DS/SR/v4/masks/Mask_Generation_v4_PLeft.png \
   --show_dashboard \
  --log_every_n 1
```

```bash
env -u DISPLAY QT_QPA_PLATFORM=cocoa pixi run check-input \
  --camera_topic /camera/back_view/image_raw \
  --front_camera_topic /camera/front_view/image_raw \
  --use_compressed \
  --safety_area PLeft PRight \
  --area_names PLeft PRight \
  --static_mask_paths \
    /Users/rashid/data/DS/SR/v4/masks/Mask_Generation_v4_PLeft.png \
    /Users/rashid/data/DS/SR/v4/masks/Mask_Generation_v4_PRight.png \
  --show_dashboard \
  --dual_camera_dashboard \
  --dashboard_width 1800 \
  --dashboard_height 1000 \
  --log_every_n 10
```

---

## Notes

- `cyclonedds-macos.xml` uses automatic network interface detection (`autodetermine="true"`), making it compatible with both Wi-Fi and Ethernet interfaces on macOS.
- `ROS_LOCALHOST_ONLY=1` is recommended when replaying bags locally. Remove it when communicating with external ROS2 machines.
- This configuration is only required on macOS. Linux users should continue using the default CycloneDDS configuration.
