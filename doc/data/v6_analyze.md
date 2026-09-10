## Analyze Normal Data -- TRAINING

```bash
python3 -c "import cv2, numpy, torch, torchvision, plotly, yaml; print('Dependencies OK')"
```

```bash
pixi run python scripts/analyze_normal_training_quality.py \
  --config configs/cf_dataset_mac.yaml \
  --safety-area ConvBelt \
  --progress
```

```bash
pixi run python scripts/analyze_normal_training_quality.py \
  --config configs/cf_dataset_mac.yaml \
  --safety-area ConvBelt \
  --embedding-samples 0 \
  --progress
```

## Wihtout Pixi

```bash
python3 scripts/analyze_normal_training_quality.py \
  --config configs/cf_dataset_mac.yaml \
  --safety-area ConvBelt \
  --progress
```

### Quick test:

```bash
pixi run python scripts/analyze_normal_training_quality.py \
  --config configs/cf_dataset_mac.yaml \
  --safety-area ConvBelt \
  --max-frames 1000 \
  --embedding-samples 500 \
  --progress
```

Output:

- Brightness and contrast
- Normalized color histograms
- Edge density
- Blur score
  Dark and saturated pixel percentages
- Foreground/mask occupancy
- Background-difference object-occupancy proxy
- Robust appearance-outlier scores
- Perceptual-hash exact duplicates
- SSIM near-duplicates
- Effective unique-frame count
- Pretrained ResNet-18 embeddings
- PCA visualization
- Semantic clustering and cluster sizes
- Embedding cosine-neighbor distance
- Most isolated samples beside their nearest-neighbor images
- Top frames requiring manual normality review
