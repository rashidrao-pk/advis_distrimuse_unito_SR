## Inference

```bash
pixi run python scripts/inference_live_shifted.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode --add_score_name   --zenoh_endpoint tcp/127.0.0.1:7447   --frame_shift_y -20   --frame_shift_fill replicate --threshold_amplification 3 3.1 3.5 8
```

## Zenoh Dashboard

```bash
python zenoh_dashboard/dashboard_viewer.py   --zenoh-endpoint tcp/127.0.0.1:7447   --camera-topic /camera/back_view/image_raw   --camera-message-type compressed   --camera-timeout 2   --inference-timeout 3
```

## Record Rosbags

```bash
pixi run bash -c '
set -e

source /home/unito/advis/distrimuse-ros2-api/install/setup.bash

new_bag="/home/unito/advis/bags/camera_detection_$(date +%Y%m%d_%H%M%S)_${RANDOM}_$$"

echo "New recording: $new_bag"
test ! -e "$new_bag"

exec ros2 bag record \
  --storage mcap \
  --output "$new_bag" \
  --topics \
  /camera/back_view/image_raw \
  /advis/detections \
  /rulex/data
'
```

- Verify rosbags

```bash
pixi run ros2 bag info \
  /Users/rashid/data/DS/SR/v6/recorded_bags/camera_detection_20261006_102510_1034_5030
```

- Preview what would be transferred without downloading:

```bash
rsync -avhn \
  HPZ3_at_SR:/home/unito/advis/bags/ \
  /Users/rashid/data/DS/SR/v6/recorded_bags/
```

## Check download prgress

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
./scripts/monitor_rosbag_transfer.sh
```
