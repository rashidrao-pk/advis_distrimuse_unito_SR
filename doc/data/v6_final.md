## Terminal 1: Publish Rosbags with `ROS messages`

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/my_git/distrimuse-image-broadcaster

pixi run replay_formatted --scenario 13_1  --loop --no-display

# or
pixi run env \
  -u CYCLONEDDS_URI \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  ROS_LOCALHOST_ONLY=1 \
  python -m cam_recorder.replay_formatted --scenario 13_1 --loop --no-display
```

## Terminal 2: Run `Zenoh` service

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
# zenohd -c zenoh_dashboard/zenoh.json5
zenohd --config zenoh_dashboard/zenoh.json5 --cfg='listen/endpoints:["tcp/0.0.0.0:7447"]'
ipconfig getifaddr en0

## CHeck from another Machine
nc -vz 172.16.97.241 7447

# pixi run python zenoh_dashboard/dashboard_viewer.py \
#   --zenoh-endpoint tcp/172.16.97.241:7447 \
#   --no-camera-monitor
```

## Terminal 3: Run Inference Live (Publishes scores and details)

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
pixi run python scripts/inference_live.py \
  --config configs/cf_dataset_mac.yaml \
  --camera_topic /camera/back_view/image_raw \
  --message_type auto \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 1.5 \
  --quantile 0.99 \
  --taas_backend cython \
  --taas_variant minimization \
  --rolling none \
  --publish_rulex \
  --rulex_topic /rulex/data \
  --detections_topic /advis/detections \
  --log_every_n 1 \
  --profile_timing \
  --publish_zenoh \
  --debug_mode \
  --zenoh_endpoint tcp/127.0.0.1:7447
```

## Terminal 4: Subscribe and see scores using timeline

### CHECK DETECTION SCORES BEING PUBLIHSED

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
# VERIFY IF ROS MESSAGES ARE BEING PUBLISHED
pixi run ros2 topic hz /rulex/data
```

### CHECK GUI BASED TIMELINE FOR DETECTION SCORES

```bash
# SEE GUI BASED TIMELINE
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
env -u DISPLAY QT_QPA_PLATFORM=cocoa \
  pixi run python zenoh_dashboard/timeline_viewer.py \
  --x-axis elapsed
```

## Terminal 5: Subscribe to Media using Dashboard

### DASHBAORD

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
env -u DISPLAY QT_QPA_PLATFORM=cocoa \
pixi run python zenoh_dashboard/dashboard_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --camera-topic /camera/back_view/image_raw \
  --camera-message-type compressed \
  --camera-timeout 2 \
  --inference-timeout 3
```

### DEBUG_VIEWER

```bash
cd /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR
env -u DISPLAY QT_QPA_PLATFORM=cocoa \
  pixi run python zenoh_dashboard/debug_viewer.py \
  --zenoh-endpoint tcp/127.0.0.1:7447 \
  --zenoh-key advis/vis/debug/state
```
