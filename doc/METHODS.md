### TAAS:

$$s(x,\hat{x}) =
Q_q\left(
G_\sigma\left(
\min_{|\Delta i|,|\Delta j|\leq o}
\lVert x_{i,j}-\hat{x}_{i+\Delta i,j+\Delta j}\rVert_2
\right)\right)$$

In practical terms:
1. Compare the original and reconstructed image.
2. For each pixel, search a spatial neighborhood controlled by offset.
3. Keep the smallest RGB distance in that neighborhood.
4. Smooth the resulting error map using sigma.
5. Use the selected upper quantile as the frame anomaly score.

`offset` Controls spatial tolerance:

```text
offset=0  exact pixel alignment
offset=1  search a 3×3 neighborhood
offset=2  search a 5×5 neighborhood
offset=3  search a 7×7 neighborhood
```

A larger offset:
- Tolerates small reconstruction shifts.
- Reduces false alarms caused by alignment errors.
- Can hide small localized anomalies if too large.
- Increases computation substantially.

`sigma` Controls Gaussian smoothing of the residual map:

```text
sigma=0    no smoothing
sigma=0.5  light smoothing
sigma=1.0  moderate smoothing
sigma=1.5  stronger smoothing
```
A larger sigma:
- Suppresses isolated noisy pixels.
- Makes scores more stable.
- Can blur small defects or thin objects.

`quantile` Controls how much of the worst residual region determines the frame score:

```text
quantile=1.0    maximum residual
quantile=0.999  worst 0.1% boundary
quantile=0.99   worst 1% boundary
quantile=0.95   worst 5% boundary
```

A high quantile is sensitive to small anomalies but also noise. A lower quantile is more robust but may miss small anomalies.