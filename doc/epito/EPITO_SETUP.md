```bash
zip /Users/rashid/data/PhD/datacloud_data/repos/XAI/ShapBPT-SAM/shap_bpt_sam/src/data/risultati_120

scp /Users/rashid/data/DS/SR/V6/Jul27/Archive.zip mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets/AD/SR/V6

# Upload SAM Results


unzip -o /beegfs/home/mrashid/datasets/AD/SR/V6/Archive.zip \
  -d /beegfs/home/mrashid/datasets/AD/SR/V6/rosbags \
  -x "__MACOSX/*"

ls -lah /beegfs/home/mrashid/datasets/AD/SR/V6/rosbags
du -sh /beegfs/home/mrashid/datasets/AD/SR/V6/rosbags

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