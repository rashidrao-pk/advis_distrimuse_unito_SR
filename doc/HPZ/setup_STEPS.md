## Step 1:

- Update repo:

```bash
ssh HPZ3_at_SR
cd ~advis_dis.....

git pull
```

## Step 2: Compile and install TAAS

```bash
conda activate dm_unito
python -m pip install cython

python scripts/setup_taas_cython.py build_ext --inplace
ls scripts/tass_cython_distance*.so

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/test_taas_cython.py -v

```

## Step 3: Upload Dataset:

```bash

cd /Users/rashid/data/DS/SR/v6/Jul27
zip -rq train.zip train

## LOCAL TO SERVER
scp unito@distrimuse.idrago.org:/home/unito/advis/DS/SR/v6/train.zip ~/Downloads/

scp /Users/rashid/data/DS/SR/v6/Jul27/train.zip \
    HPZ3_at_SR:/home/unito/advis/DS/SR/v6/

ls -lh /home/unito/advis/DS/SR/v6/train.zip

unzip -o /home/unito/advis/DS/SR/v6/train.zip \
  -d /home/unito/advis/DS/SR/v6/train \
  -x "__MACOSX/*"

ls -lah /home/unito/advis/DS/SR/v6/train
du -sh /home/unito/advis/DS/SR/v6/train


```

## Step 3: Check Model Checkpoints

```bash
python scripts/check_model_checkpoints.py \
 --config configs/cf_dataset_hp.yaml
```

### Download checkpoints:

```bash
pixi run hf --version

mkdir -p results/V6/train/models

pixi run hf download rashidrao/ADVIS_SR_DISTRIMUSE \
  --include "checkpoints/*" \
  --local-dir /tmp/advis_hf

mv /tmp/advis_hf/checkpoints/* results/V6/train/models/
rm -rf /tmp/advis_hf

```

```bash
python scripts/check_model_checkpoints.py \
 --config configs/cf_dataset_hp.yaml

python scripts/check_model_checkpoints.py \
--config configs/cf_dataset_hp.yaml \
--json
```

## Dummy Inference Test

```bash
python scripts/test_model_inference.py \
  --config configs/cf_dataset_hp.yaml \
  --data_source training \
  --safety_area PLeft PRight RoboArm ConvBelt \
  --max_images 32
```

### Download Thresholds

```bash
mkdir -p results/V6

pixi run hf download rashidrao/ADVIS_SR_DISTRIMUSE \
  --include "thresholds/**" \
  --local-dir results/V6

# verify
find results/V6/thresholds -maxdepth 2 -type f \
  -name '*percentile99.0_off1_sig1.0_q0.99*.json' | sort
```

### Inference:

```bash
conda actiavte dm_unito

#  RUN OFFLINE USING SAVED VIDEO
python scripts/infer_offline.py   --config configs/cf_dataset_hp.yaml   --input_type video   --scenario 13_1   --topic /camera/back_view/image_raw   --safety_areas ALL   --threshold_strategy percentile   --offset 1   --sigma 1.0   --quantile 0.99 --scores-only

#  RUN OFFLINE USING SAVED ROSBAG
pixi run python scripts/infer_offline.py \
  --config configs/cf_dataset_hp.yaml \
  --input_type rosbag \
  --scenario 8_0 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --max_frames 200 \
  --skip-first 100

```

- --threshold_amplification
  - 1.05 1.05 1.05 1.05
  - PLeft PRight RoboArm ConvBelt

```bash
pixi run python scripts/inference_live.py \
  --config configs/cf_dataset_hp.yaml \
  --camera_topic /camera/back_view/image_raw \
  --message_type auto \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --threshold_amplification 1.1 1.1 1.1 1.2 \
  --offset 1 \
  --sigma 1.0 \
  --quantile 0.99 \
  --taas_backend auto \
  --rolling mean \
  --rolling_window 5 \
  --detections_topic /advis/detections \
  --log_every_n 1 \
  --profile_timing \
  --publish_zenoh \
  --debug_mode \
  --zenoh_endpoint tcp/127.0.0.1:7447
  --publish_rulex \
  --rulex_topic /rulex/data \
```

## Run DASHBOARD VIEWER

```bash
python zenoh_dashboard/dashboard_viewer.py   --zenoh-endpoint tcp/127.0.0.1:7447   --camera-topic /camera/back_view/image_raw   --camera-message-type compressed   --camera-timeout 2   --inference-timeout 3
```

---

## TEST AT 5th Oct 14:01

```bash
pixi run python scripts/inference_live.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --threshold_amplification 1.5 1.5 1.5 1.5   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode   --zenoh_endpoint tcp/127.0.0.1:7447
```

```bash
pixi run python scripts/inference_live.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --threshold_amplification 3.5 3.5 6.5 5.5   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode   --zenoh_endpoint tcp/127.0.0.1:7447
```

```bash
cd ~/advis/advis_distrimuse_unito_SR

pixi run ros2 topic type /camera/back_view/image_raw
pixi run ros2 topic hz /camera/back_view/image_raw
```

```bash
mkdir -p /home/unito/advis/bags

bag_path="/home/unito/advis/bags/camera_$(date +%Y%m%d_%H%M%S)"

pixi run ros2 bag record \
  --storage mcap \
  --output "$bag_path" \
  /camera/back_view/image_raw
```

```bash
pixi run ros2 bag info "$bag_path"
```

```bash
pixi run ros2 bag play "$bag_path" \
  --topics /camera/back_view/image_raw
```

## Check RuleX Detection

```bash
pixi run bash -c '
source /home/unito/advis/distrimuse-ros2-api/install/setup.bash
ros2 topic hz /rulex/data'
```

## Recording Rosbags and RuleX detections

```bash
bag_path="/home/unito/advis/bags/camera_detection_$(date +%Y%m%d_%H%M%S)"
echo "$bag_path"

pixi run bash -c "
source /home/unito/advis/distrimuse-ros2-api/install/setup.bash

ros2 bag record \
  --storage mcap \
  --output '$bag_path' \
  --topics \
  /camera/back_view/image_raw \
  /advis/detections \
  /rulex/data
"
# /home/unito/advis/bags/camera_detection_20261005_143503

df -h /home/unito/advis/bags

du -sh /home/unito/advis/bags

du -sh /home/unito/advis/bags/* 2>/dev/null | sort -hr


```

##

```bash
pixi run python scripts/inference_live.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --threshold_amplification 4 3.5 3.5 7.5   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode   --zenoh_endpoint tcp/127.0.0.1:7447
```

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

```bash
pixi run python scripts/inference_live.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --threshold_amplification 4 3.5 3.5 8   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode   --zenoh_endpoint tcp/127.0.0.1:7447
```

```bash
BAG_PATH="$latest_bag" pixi run python - <<'PY'
import os
from pathlib import Path
import yaml

bag = Path(os.environ["BAG_PATH"])
metadata = yaml.safe_load((bag / "metadata.yaml").read_text())
info = metadata["rosbag2_bagfile_information"]
topics = info["topics_with_message_count"]

duration = info["duration"]["nanoseconds"] / 1e9

print(f"Bag: {bag}")
print(f"Duration: {duration:.2f} seconds")
print(f"Total messages: {info['message_count']}")
print(f"Topics: {len(topics)}")
print()

for item in topics:
    topic = item["topic_metadata"]
    count = int(item["message_count"])
    rate = count / duration if duration > 0 else 0
    kind = "frames" if topic["name"] == "/camera/back_view/image_raw" else "messages"
    print(
        f"{topic['name']}\n"
        f"  type: {topic['type']}\n"
        f"  {kind}: {count}\n"
        f"  average rate: {rate:.2f} Hz"
    )
PY
```

### Recorded Rosbags:

> New recording: /home/unito/advis/bags/camera_detection_20261006_104650_4369_6007

- Back

> New recording: /home/unito/advis/bags/camera_detection_20261006_110422_25125_7410

- Another

  > New recording: /home/unito/advis/bags/camera_detection_20261006_111548_15274_7637

- Anoteher

  > New recording: /home/unito/advis/bags/camera_detection_20261006_112332_23270_7719

- Anoteher
  > New recording: /home/unito/advis/bags/camera_detection_20261006_112721_6439_7786

```bash
pixi run python scripts/inference_live.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile  --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode   --zenoh_endpoint tcp/127.0.0.1:7447 --threshold_amplification 4.2 3.5 3.5 8
```

```bash
pixi run python scripts/inference_live_shifted.py \
  --config configs/cf_dataset_hp.yaml \
  --camera_topic /camera/back_view/image_raw \
  --message_type auto \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 2.0 \
  --quantile 0.99 \
  --taas_backend cython \
  --rolling min \
  --rolling_window 10 \
  --detections_topic /advis/detections \
  --log_every_n 1 \
  --profile_timing \
  --publish_zenoh \
  --debug_mode \
  --zenoh_endpoint tcp/127.0.0.1:7447 \
  --threshold_amplification 4.2 3.5 3.5 8 \
  --frame_shift_y -20 \
  --frame_shift_fill replicate
```

```bash
pixi run python scripts/inference_live_shifted.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode   --zenoh_endpoint tcp/127.0.0.1:7447   --frame_shift_y -20   --frame_shift_fill replicate --threshold_amplification 3 3 3.5 8
```

```bash
pixi run python scripts/inference_live_shifted.py   --config configs/cf_dataset_hp.yaml   --camera_topic /camera/back_view/image_raw   --message_type auto   --safety_areas ALL   --threshold_strategy percentile   --offset 3   --sigma 2.0   --quantile 0.99   --taas_backend cython   --rolling min   --rolling_window 10   --detections_topic /advis/detections   --log_every_n 1   --profile_timing   --publish_zenoh   --debug_mode --add_score_name   --zenoh_endpoint tcp/127.0.0.1:7447   --frame_shift_y -20   --frame_shift_fill replicate --threshold_amplification 3 3.1 3.5 8
```
