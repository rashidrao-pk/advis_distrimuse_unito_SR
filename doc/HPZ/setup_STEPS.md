
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


```bash
python scripts/test_model_inference.py \
  --config configs/cf_dataset_hp.yaml \
  --data_source training \
  --safety_area PLeft PRight RoboArm ConvBelt \
  --max_images 32
```

