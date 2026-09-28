
## Step 1:
 - Update repo:
```bash
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