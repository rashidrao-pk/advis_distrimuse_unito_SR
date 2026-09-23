```bash
# Check available ROS Messages
pixi run ros2 topic list

pixi run ros2 topic list -t

pixi run ros2 interface list | grep "/msg/"

pixi run ros2 interface show sensor_msgs/msg/Image

## CHECK IF CAMERA IS ALIVE
pixi run ros2 topic info /camera/back_view/image_raw -v

pixi run ros2 interface show distrimuse_ros2_api/msg/RulexDetectionResult

## Check msgs continously
pixi run ros2 topic hz /camera/back_view/image_raw

pixi run ros2 topic echo /rulex/data

pixi run ros2 topic echo /advis/detections

```

### Check Publised from UniTo/RuleX:

```bash
pixi run ros2 topic info /rulex/data --verbose
pixi run ros2 topic type /advis/dashboard/compressed
pixi run ros2 topic echo /rulex/data --once
pixi run ros2 topic hz /rulex/data
pixi run ros2 topic echo /rulex/data --field area_scores --once
```

```bash
pixi run ros2 topic echo /advis/dashboard/compressed --once --no-arr
```

## Check subscriptions of topics

```bash
pixi run ros2 topic info /camera/back_view/image_raw -v
pixi run ros2 topic info /advis/detections -v
pixi run ros2 topic info /rulex/data -v
```
