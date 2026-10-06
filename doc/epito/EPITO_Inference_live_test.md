```bash

srun -p epito --gres=gpu:a100:1 -J "ADVIS-Cal" --pty bash
tmux new -s AD_SR_Cal
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
cd /beegfs/home/mrashid/repos/advis_distrimuse_unito_SR

#
python scripts/infer_offline.py   --config configs/cf_dataset_epito.yaml   --input_type video   --scenario 8_16   --topic /camera/back_view/image_raw   --safety_areas ALL   --threshold_strategy percentile   --offset 3   --sigma 2.0   --quantile 0.99 --rolling min --threshold_amplification 1.05 --profile_timing
```

## Run for User Study - XAI User Study

```bash
python scripts/infer_offline.py   --config configs/cf_dataset_epito.yaml   --input_type video   --scenario 13_0   --topic /camera/back_view/image_raw   --safety_areas ALL   --threshold_strategy percentile   --offset 3   --sigma 2.0   --quantile 0.999 --rolling min --threshold_amplification 1.05 --profile_timing


#  13_1
python scripts/infer_offline_study.py --config configs/cf_dataset_epito.yaml --input_type video --scenario 13_1 --topic /camera/back_view/image_raw --safety_areas ALL --threshold_calibration_mode val --threshold_strategy percentile --offset 3 --sigma 2.0 --quantile 0.99 --taas_backend cython --add_score_name --rolling min --threshold_amplification 1.5 1.15 1.2 1.2 --profile_timing
```

```bash
#  12_0 --> Unexpected Box on Conveyor Belt
python scripts/infer_offline_study.py --config configs/cf_dataset_epito.yaml --input_type video --scenario 12_0 --topic /camera/back_view/image_raw --safety_areas ALL --threshold_calibration_mode val --threshold_strategy percentile --offset 3 --sigma 2.0 --quantile 0.99 --taas_backend cython --add_score_name --rolling min --threshold_amplification 1.10 1.10 1.1 1.1
```
