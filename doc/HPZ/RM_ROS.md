# ROS monitoring and recovery on HP

> HPZ documentation: [First-time setup](SETUP.md) · [Daily run](RUN_steps.md) ·
> **ROS monitoring and recovery** · [System services](SERVICES.md)

Use this guide when the camera, inference, dashboard, or Rulex messages are
missing or stale. Complete [SETUP.md](SETUP.md) before the first run and use
[RUN_steps.md](RUN_steps.md) for the normal startup sequence.

Run commands from the ADVIS repository unless a section says otherwise:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
```

## 1. Verify the ROS environment

All terminals participating in ROS discovery must use the same domain and
compatible CycloneDDS configuration.

```bash
pixi run python -c \
  "import os; print('RMW=', os.environ.get('RMW_IMPLEMENTATION')); print('DDS=', os.environ.get('CYCLONEDDS_URI')); print('DOMAIN=', os.environ.get('ROS_DOMAIN_ID'))"
```

Expected HP values include:

```text
RMW= rmw_cyclonedds_cpp
DDS= /home/unito/dm/distrimuse-seds/cyclonedds.xml
DOMAIN= 1
```

Confirm the ROS Python environment itself is available:

```bash
pixi run python -c \
  "import rclpy, rosbag2_py; print('ROS Python: OK')"
```

If `rclpy` is missing, the command was probably run with another project's
Python. Use `pixi run`, not bare `python`.

## 2. Inspect discovered topics and message types

```bash
pixi run ros2 topic list
pixi run ros2 topic list -t
```

Useful interface checks:

```bash
pixi run ros2 interface show sensor_msgs/msg/Image
pixi run ros2 interface show sensor_msgs/msg/CompressedImage
pixi run ros2 interface show distrimuse_ros2_api/msg/RulexDetectionResult
```

If the Rulex interface is unknown, build/source the DistriMuSe ROS 2 API in
the environment used by the publisher. See the API repository instructions
before starting ADVIS with `--publish_rulex`.

## 3. Determine which component has stopped

Check the camera first:

```bash
pixi run ros2 topic type /camera/back_view/image_raw
pixi run ros2 topic info /camera/back_view/image_raw --verbose
pixi run ros2 topic hz /camera/back_view/image_raw
```

Then check inference outputs:

```bash
pixi run ros2 topic info /advis/detections --verbose
pixi run ros2 topic hz /advis/detections

pixi run ros2 topic info /rulex/data --verbose
pixi run ros2 topic hz /rulex/data
```

Interpret the result:

| Camera | Detection output | Meaning |
|---|---|---|
| Receiving frames | Receiving messages | Camera and inference are running |
| Receiving frames | Missing/stale | Camera is alive; inference stopped or disconnected |
| Missing/stale | Missing/stale | Camera publisher, VLAN, or ROS discovery is unavailable |
| Missing/stale | Last detection remains visible | Dashboard is displaying cached inference state |

The dashboard timeout should eventually distinguish `CAMERA OFFLINE` from
`INFERENCE STOPPED`. Topic checks above provide the authoritative diagnosis.

## 4. Inspect messages without flooding the terminal

Read one detection payload:

```bash
pixi run ros2 topic echo /advis/detections --once
```

Read one Rulex message and its area scores:

```bash
pixi run ros2 topic echo /rulex/data --once
pixi run ros2 topic echo /rulex/data --field area_scores --once
```

Check the dashboard media topic without printing the binary image array:

```bash
pixi run ros2 topic type /advis/dashboard/compressed
pixi run ros2 topic info /advis/dashboard/compressed --verbose
pixi run ros2 topic hz /advis/dashboard/compressed
pixi run ros2 topic echo /advis/dashboard/compressed --once --no-arr
```

## 5. Verify publisher/subscriber connections

```bash
pixi run ros2 topic info /camera/back_view/image_raw --verbose
pixi run ros2 topic info /advis/detections --verbose
pixi run ros2 topic info /advis/dashboard/compressed --verbose
pixi run ros2 topic info /rulex/data --verbose
```

For each topic, inspect:

- topic type;
- publisher count;
- subscriber count;
- node names;
- QoS compatibility.

A topic name can exist while having zero active publishers, so `topic list`
alone does not prove that frames or detections are flowing.

## 6. Check VLAN and network state

```bash
ip -brief address
ip route
```

Review the configured CycloneDDS interfaces and peers:

```bash
sed -n '1,220p' /home/unito/dm/distrimuse-seds/cyclonedds.xml
```

If VLAN setup is managed by systemd:

```bash
systemctl status vlan_setup.service --no-pager
sudo journalctl -u vlan_setup.service -n 100 --no-pager
```

For a manual VLAN restart, follow Step 3 in
[RUN_steps.md](RUN_steps.md#3-prepare-the-seds-vlan-and-ros-network).

## 7. Refresh ROS discovery safely

Restarting the ROS daemon does not stop camera or inference nodes. It refreshes
the CLI discovery cache:

```bash
pixi run ros2 daemon stop
pixi run ros2 daemon start
pixi run ros2 node list
pixi run ros2 topic list -t
```

If messages still do not flow, restart the affected publisher/subscriber rather
than repeatedly restarting the daemon.

For service-managed inference:

```bash
sudo systemctl restart inference.service
sudo journalctl -u inference.service -n 100 --no-pager
```

For manual inference, stop it with `Ctrl-C` and restart it using the command in
[RUN_steps.md](RUN_steps.md#terminal-b-live-inference).

## 8. Detect duplicate or stale processes

```bash
pgrep -af 'inference_live.py|infer_ros_live|dashboard_viewer.py|zenohd|ros2 bag play'
```

Do not run a manual inference process while `inference.service` is active.
Prefer a graceful `Ctrl-C` for manual programs or `systemctl stop` for services.

```bash
systemctl is-active inference.service
sudo systemctl stop inference.service
```

After cleanup, verify that only the intended processes remain:

```bash
pgrep -af 'inference_live.py|dashboard_viewer.py|zenohd|ros2 bag play'
```

## 9. Check Zenoh when the ROS topics work but the dashboard is empty

Confirm that the router is listening:

```bash
ss -ltnp | grep 7447
pgrep -af zenohd
```

Start it manually if needed:

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR
zenohd -c zenoh_dashboard/zenoh.json5
```

The inference process and viewer must use the same endpoint:

```text
tcp/127.0.0.1:7447
```

When connecting from another PC, the router must listen on a network-accessible
address, the firewall must allow TCP port 7447, and the remote viewer must use
the HP machine's IP rather than `127.0.0.1`.

## 10. Recommended recovery order

Use this order to avoid restarting working components unnecessarily:

1. Confirm `ROS_DOMAIN_ID`, RMW implementation, and CycloneDDS URI.
2. Confirm VLAN interface/address and routes.
3. Check camera publisher count and frame rate.
4. Check inference process/service and its logs.
5. Check `/advis/detections` and `/rulex/data` rates.
6. Check the Zenoh router and TCP port 7447.
7. Restart only the failed component.
8. Restart the ROS daemon only to refresh CLI discovery.

For service commands and persistent logs, continue with
[SERVICES.md](SERVICES.md).
