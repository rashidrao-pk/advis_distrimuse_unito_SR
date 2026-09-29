## Camera offline? Run Self Broadcast

- This runs a `custom replay function` which has support for `collected sceanrios`
  > cam_recorder.`replay_formatted`

```bash
cd /home/unito/advis/advis_support/distrimuse-image-broadcaster
pixi run env   -u CYCLONEDDS_URI   RMW_IMPLEMENTATION=rmw_fastrtps_cpp   ROS_LOCALHOST_ONLY=1   python -m cam_recorder.replay_formatted --scenario 13_0 --loop --no-display
```
