# ADVIS split cleanup status

## Deliverables created

- `infer_ros_live_zenoh_standalone.py`
- `dashboard_viewer_standalone.py`
- `timeline_viewer_standalone.py`

## What changed

### Inference application
- Removed local dashboard and timeline display flow.
- Removed GUI CLI flags from inference.
- Added direct Zenoh publishing for:
  - `advis/vis/dashboard/state`
  - `advis/vis/timeline/state`
- Inlined the payload packing helpers so the producer is self-contained.
- Trimmed producer visualization state to transport-only fields:
  - `bbox`
  - `mask_bin`
  - `resize_meta`
  - image patches added after inference
- Added Zenoh publisher/session cleanup on shutdown.

### Dashboard viewer
- Flattened into a single runnable file.
- Inlined decode/render/mask helper logic.
- Removed package-relative imports.
- Keeps "get latest stored state" + live subscription behavior.

### Timeline viewer
- Flattened into a single runnable file.
- Inlined timeline decode/render logic.
- Removed package-relative imports.
- Keeps "get latest stored state" + live subscription behavior.

## Remaining validation to do on target environment

1. Start `zenohd` with the desired storage configuration.
2. Run the producer and confirm dashboard/timeline messages are published.
3. Launch each viewer independently on another machine.
4. Confirm a late-started viewer immediately renders the last stored message.
5. Confirm image bandwidth is acceptable with current JPEG quality (`--zenoh-jpeg-quality`, default `85`).
6. Confirm the local environment has all runtime dependencies installed:
   - producer: ROS2 deps, torch, torchvision, opencv, matplotlib, msgpack, eclipse-zenoh
   - viewers: opencv, numpy, msgpack, eclipse-zenoh

## Suggested runtime commands

### Producer
```bash
python3 infer_ros_live_zenoh_standalone.py \
  --static_mask_paths <mask1> <mask2> <mask3> <mask4> \
  --threshold_dir <threshold_dir>
```

### Dashboard viewer
```bash
python3 dashboard_viewer_standalone.py --zenoh-endpoint tcp/127.0.0.1:7447
```

### Timeline viewer
```bash
python3 timeline_viewer_standalone.py --zenoh-endpoint tcp/127.0.0.1:7447
```
