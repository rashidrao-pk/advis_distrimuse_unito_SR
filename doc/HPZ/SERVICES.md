# ADVIS system services on HP

> HPZ documentation: [First-time setup](SETUP.md) · [Daily run](RUN_steps.md) ·
> [ROS monitoring and recovery](RM_ROS.md) · **System services**

Use systemd for unattended or boot-time VLAN and inference startup. For
interactive development, stop `inference.service` and follow the manual flow in
[RUN_steps.md](RUN_steps.md). Do not run manual and service instances of
inference simultaneously.

## 1. Check service state

```bash
systemctl status vlan_setup.service inference.service --no-pager
systemctl is-enabled vlan_setup.service inference.service
systemctl is-active vlan_setup.service inference.service
```

For a compact failed-unit overview:

```bash
systemctl --failed
```

## 2. Start, stop, and restart

Start in dependency order:

```bash
sudo systemctl start vlan_setup.service
sudo systemctl start inference.service
```

Stop inference before the network service:

```bash
sudo systemctl stop inference.service
sudo systemctl stop vlan_setup.service
```

Restart inference after changing models, thresholds, configuration, or code:

```bash
sudo systemctl restart inference.service
```

Restart the complete runtime stack:

```bash
sudo systemctl restart vlan_setup.service
sudo systemctl restart inference.service
```

## 3. Control automatic startup

Enable and immediately start both services:

```bash
sudo systemctl enable --now vlan_setup.service
sudo systemctl enable --now inference.service
```

Stop and disable automatic startup:

```bash
sudo systemctl disable --now inference.service
sudo systemctl disable --now vlan_setup.service
```

Enable at boot without starting immediately:

```bash
sudo systemctl enable vlan_setup.service inference.service
```

## 4. View logs

Follow live inference logs:

```bash
sudo journalctl -u inference.service -f
```

Show recent inference and VLAN logs:

```bash
sudo journalctl -u inference.service -n 200 --no-pager
sudo journalctl -u vlan_setup.service -n 200 --no-pager
```

Show logs from the current boot:

```bash
sudo journalctl -b -u vlan_setup.service -u inference.service --no-pager
```

## 5. Inspect the effective service configuration

```bash
systemctl cat vlan_setup.service
systemctl cat inference.service

systemctl show inference.service \
  -p WorkingDirectory \
  -p ExecStart \
  -p Environment \
  -p User
```

Confirm that `inference.service` uses:

- `/home/unito/advis/advis_distrimuse_unito_SR` as its working directory;
- this repository's Pixi environment or `pixi run` in `ExecStart`;
- `ROS_DOMAIN_ID=1`;
- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`;
- the HP CycloneDDS configuration;
- `configs/cf_dataset_hp.yaml`.

If the service uses bare `python`, it may select the wrong environment and
fail to import `rclpy`, `rosbag2_py`, or Cython extensions.

## 6. Apply unit-file changes

After editing a `.service` file:

```bash
sudo systemctl daemon-reload
sudo systemctl restart vlan_setup.service
sudo systemctl restart inference.service
```

Then verify:

```bash
systemctl status vlan_setup.service inference.service --no-pager
```

## 7. Update code and dependencies used by the service

Stop inference before updating:

```bash
sudo systemctl stop inference.service

cd /home/unito/advis/advis_distrimuse_unito_SR
git pull
pixi install --locked
pixi run python scripts/setup_taas_cython.py build_ext --inplace
```

Run a short preflight:

```bash
pixi run python -c \
  "import rclpy, rosbag2_py, torch; print('CUDA:', torch.cuda.is_available())"

PYTHONPATH=scripts pixi run python -c \
  "import tass_cython_distance; print(tass_cython_distance.__file__)"
```

Start inference again:

```bash
sudo systemctl start inference.service
sudo journalctl -u inference.service -f
```

## 8. Verify runtime topics after service startup

```bash
cd /home/unito/advis/advis_distrimuse_unito_SR

pixi run ros2 topic hz /camera/back_view/image_raw
pixi run ros2 topic hz /advis/detections
pixi run ros2 topic hz /rulex/data
```

If the camera is live but detection topics are stale, inspect inference logs.
If both camera and inference topics are missing, inspect the VLAN service and
follow [RM_ROS.md](RM_ROS.md).

## 9. Return to service mode after a manual test

After stopping all manually launched inference/dashboard/router processes:

```bash
pgrep -af 'inference_live.py|dashboard_viewer.py|zenohd'

sudo systemctl enable --now vlan_setup.service
sudo systemctl enable --now inference.service
```

Confirm that only the intended service-managed inference instance is running:

```bash
systemctl is-active inference.service
sudo journalctl -u inference.service -n 50 --no-pager
```
