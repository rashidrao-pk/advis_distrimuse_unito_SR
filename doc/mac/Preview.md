```bash
pixi run python scripts/preview_r2ad_mask.py \
  --image /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames/1_0/back_view/raw/frame_000000.png \
  --mask /Users/rashid/data/DS/SR/v6/masks/Mask_Generation_v6_SafetyZone.png \
  --output-dir reports/r2ad_mask_preview_1_0_random \
  --sampling random \
  --num-patches 48 \
  --seed 42
```
