```bash
zip /Users/rashid/data/DS/SR/v6/Jul27/test

scp /Users/rashid/data/DS/SR/v6/Jul27/test.zip mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets/AD/SR/V6


# Upload test data

unzip -o /beegfs/home/mrashid/datasets/AD/SR/V6/test.zip \
  -d /beegfs/home/mrashid/datasets/AD/SR/V6/test \
  -x "__MACOSX/*"

ls -lah /beegfs/home/mrashid/datasets/AD/SR/V6/test
du -sh /beegfs/home/mrashid/datasets/AD/SR/V6/test

```

## Install Cython and TAAS

[Follow this](/doc/METHODS.md#taas)

```bash

conda activate pt_312
python -m pip install cython
```

```bash
python scripts/setup_taas_cython.py build_ext --inplace
ls scripts/tass_cython_distance*.so
```

```text
| /ROSBAGS
|------ extracted_frames
|---------- 1_0
|-------------- back_view
|----------------- processed
|---------------------- PLeft
|---------------------- PRight
|---------------------- ConvBelt
|---------------------- RoboArm
|-------------- 1_0_back_view_masked.mp4
|--------------
|---------- 2_0
|-------------
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|-------------
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24
|---------- Jul27_Scenario_1_0_2026-07-27_10-25-24

```
