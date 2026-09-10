## Annotate Safety Areas

Features included:

- Frame slider and Previous/Next controls
- Automatic playback at selectable speeds
- Safety-area selector
- Raw and processed frame views when raw frames exist
- Label a single frame or an inclusive frame sequence
- Normal, Anomalous, and Unlabeled states
- Optional annotation notes
- Keyboard shortcuts:
  - Left/Right arrows: navigate
  - N: label Normal
  - A: label Anomalous
- Browser autosave using local storage
- Import an existing annotation CSV
- Download a full-scenario CSV containing every frame
- Scenario description loaded from the YAML
- Dataset images referenced by path rather than embedded in the HTML

```bash
pixi run python scripts/annotate_safety_area.py \
  /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames/1_0 \
  --camera back_view
```

```bash
pixi run python scripts/annotate_safety_area.py \
  /Users/rashid/data/DS/SR/v6/Jul27/extracted_frames/1_0 \
  --camera back_view \
  --areas PLeft PRight
```
