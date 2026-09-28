
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

## Step 3: Check Model Checkpoints

```bash
python scripts/check_model_checkpoints.py \
 --config configs/cf_dataset_hp.yaml
```