### Verify Downloaded Rosbags

```bash
base="/Users/rashid/data/DS/SR/v6/recorded_bags"

for bag_dir in "$base"/*; do
  [[ -d "$bag_dir" ]] || continue

  echo
  echo "============================================================"
  echo "$(basename "$bag_dir")"
  echo "============================================================"

  if [[ ! -f "$bag_dir/metadata.yaml" ]]; then
    echo "❌ Invalid/incomplete: metadata.yaml missing"
    continue
  fi

  if ! find "$bag_dir" -maxdepth 1 -name "*.mcap" -type f | grep -q .; then
    echo "❌ Invalid/incomplete: MCAP missing"
    continue
  fi

  pixi run ros2 bag info "$bag_dir" \
    && echo "✅ Bag metadata and MCAP are readable" \
    || echo "❌ ros2 bag info failed"
done
```
