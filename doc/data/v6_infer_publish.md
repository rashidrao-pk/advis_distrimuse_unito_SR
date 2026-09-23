## Publish Messages for Smart Robotics

- [https://github.com/smart-robotics/**distrimuse-ros2-api**/blob/main/msg/**RulexAreaScore.msg**](https://github.com/smart-robotics/distrimuse-ros2-api/blob/main/msg/RulexAreaScore.msg)

```bash
# Information about a specific area, provided by Rulex/UniTo
string AREA_A = AREA_A
string AREA_B = AREA_B
string AREA_C = AREA_C
string AREA_D = AREA_D

string area  # Name of the area (i.e. AREA_A, AREA_B, AREA_C, AREA_D)
bool anomaly  # boolean whether there is an anomaly detected or not
```

- [https://github.com/smart-robotics/**distrimuse-ros2-api**/blob/main/msg/**RulexDetectionResult.msg**](https://github.com/smart-robotics/distrimuse-ros2-api/blob/main/msg/RulexDetectionResult.msg)

```bash
# Information of what's detected in all area's, provided by Rulex/UniTo

distrimuse_ros2_api/RulexAreaScore[] area_scores  # List of detections per area
sensor_msgs/Image image # Optional image in case of anomaly
```

### UniTo safety-area mapping

The API defines generic area constants. UniTo publishes the physical safety
areas with this integration mapping:

| Safety area | Rulex API value |
| ----------- | --------------- |
| `PRight`    | `AREA_A`        |
| `PLeft`     | `AREA_B`        |
| `RoboArm`   | `AREA_C`        |
| `ConvBelt`  | `AREA_D`        |

```bash
git clone git@github.com:rashidrao-pk/distrimuse-ros2-api.git
git stash push -m "lockfile before macOS branch" -- pixi.lock

git remote add fork \
  https://github.com/rashidrao-pk/distrimuse-ros2-api.git

git fetch fork
git switch --track -c test_mac fork/test_mac

pixi install
pixi run test

pixi run colcon build --symlink-install

source install/setup.zsh

```

## Verify API:

```bash
pixi run python -c \
  "from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult; print(RulexDetectionResult())"
```

### Verify API for RuleX/UniTo

```bash
cd ~/DistriMuSe/advis_distrimuse_unito_SR
pixi add ros-kilted-distrimuse-ros2-api \
  --git https://github.com/rashidrao-pk/distrimuse-ros2-api.git \
  --branch test_mac
```

```bash
pixi run python -c \
  "from distrimuse_ros2_api.msg import RulexAreaScore, RulexDetectionResult; print('Rulex API ready')"

pixi run ros2 interface show \
  distrimuse_ros2_api/msg/RulexDetectionResult
```
