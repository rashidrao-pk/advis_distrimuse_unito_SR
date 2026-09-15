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

## Supported reconstruction anomaly scores

Let \(x\in[0,1]^{C\times H\times W}\) be an input image, \(\hat{x}\) its
reconstruction, and \(r=x-\hat{x}\) the residual. Larger values mean that the
input is less consistent with the learned normal appearance. Every score needs
its own calibrated threshold: numerical thresholds cannot be transferred from
one score family to another.

| Score name in ablation | Definition | Strength | Main limitation |
|---|---|---|---|
| `L1_mean` | Mean absolute residual | Robust, simple and comparable across equal preprocessing | Treats every pixel equally and is sensitive to small spatial shifts |
| `L2_norm` | Euclidean norm of the complete residual | Emphasizes larger errors | Depends on image dimensions and can be dominated by broad reconstruction error |
| `MSE_mean` | Mean squared residual | Standard VAE reconstruction metric | Can emphasize large pixel errors and background reconstruction artifacts |
| `RAVI_max_abs` | Maximum absolute residual per image | Very sensitive to tiny/local anomalies | Extremely sensitive to one noisy or saturated pixel |
| `dSSIM_sig*` | Structural dissimilarity, (1-SSIM) | Responds to structural/texture changes and is less tied to exact color error | Window and Gaussian sigma affect sensitivity; global averaging may dilute small anomalies |
| `TAAS_off*_sig*_q*` | Tolerance-aware anomaly score described above | Tolerates small spatial reconstruction shifts and permits tail control | Large tolerance or strong smoothing can hide small anomalies |

### L1: mean absolute reconstruction error

\[
s_{L1}(x,\hat{x})=\frac{1}{CHW}\sum_{c,i,j}|x_{cij}-\hat{x}_{cij}|
\]

L1 is a useful baseline because it has a direct interpretation as average
absolute pixel error. It is normally less dominated by a few large residuals
than MSE or L2.

### L2: full-image Euclidean residual

\[
s_{L2}(x,\hat{x})=\sqrt{\sum_{c,i,j}(x_{cij}-\hat{x}_{cij})^2}
\]

This matches `ComputeDifferences.get_l2_difference` in `scripts/utils.py`.
Unlike MSE, it is not divided by the number of pixels. Consequently, L2
thresholds are valid only for the same image size and channel configuration.

### MSE: mean squared reconstruction error

\[
s_{MSE}(x,\hat{x})=\frac{1}{CHW}\sum_{c,i,j}(x_{cij}-\hat{x}_{cij})^2
\]

MSE is included because it is the reconstruction objective used by the model.
It is useful as a reference, although the score that best matches the training
loss is not necessarily the score that best separates anomalies.

### RAVI: maximum absolute residual

For threshold ablation, RAVI is calculated per image:

\[
s_{RAVI}(x,\hat{x})=\max_{c,i,j}|x_{cij}-\hat{x}_{cij}|
\]

This follows the residual and maximum operation in
`ComputeDifferences.get_ravi_difference`. The legacy helper takes one maximum
over the complete batch. The ablation deliberately calculates a separate
maximum for every image; otherwise a frame's score would depend on which other
frames happened to be in its batch.

RAVI is appropriate when a very small, strong defect must be detected. It is
usually unstable in the presence of sensor noise, dead pixels, mask borders or
saturation.

### d-SSIM: structural dissimilarity

\[
s_{dSSIM}(x,\hat{x})=1-SSIM(x,\hat{x})
\]

SSIM close to 1 means similar structure, so d-SSIM close to 0 is normal. The
ablation compares Gaussian SSIM windows using the values passed through
`--dssim_sigmas`. d-SSIM is useful for changed shape, texture or local
structure that may not create a large mean pixel error.

## Threshold functions

For a collection of normal calibration scores \(S=\{s_1,\ldots,s_N\}\), the
following threshold functions are supported:

### Maximum normal score

\[
\tau=\max(S)
\]

This guarantees zero exceedances on the calibration set, but one unusual normal
frame can make the threshold excessively high.

### Normal-score percentile

\[
\tau=Q_p(S)
\]

For example, `--threshold_percentiles 99` uses the 99th percentile. This is
usually more robust than the maximum and directly controls the intended normal
tail probability.

### Mean plus standard deviations

\[
\tau=\operatorname{mean}(S)+k\operatorname{std}(S)
\]

This is convenient for approximately symmetric score distributions. Anomaly
scores are commonly right-skewed, so percentile calibration is generally safer
unless the distribution has been inspected.

## Normal-only threshold ablation

Run:

```bash
python scripts/ablate_validation_thresholds.py \
  --config configs/cf_dataset_epito.yaml \
  --safety_area PLeft \
  --dataset_version V6
```

The default ablation compares:

- L1, L2, MSE and per-image RAVI;
- d-SSIM with Gaussian sigma 1.0 and 1.5;
- TAAS offsets 0, 1, 2 and 3;
- TAAS smoothing sigmas 0, 0.5, 1.0 and 1.5;
- TAAS residual quantiles 0.95, 0.99, 0.999 and 1.0;
- maximum, percentile and mean-plus-standard-deviation thresholds.

The normal validation samples are alternated between calibration and evaluation
subsets. Thresholds are fitted on calibration scores, while normal false-positive
rate is measured on the evaluation scores. The report ranks stable normal
operating points using target-FPR error and the calibration/evaluation FPR gap.

Outputs:

```text
results/V6/threshold_ablation/PLeft/
├── threshold_ablation.html
├── threshold_ablation.csv
├── validation_scores_all_methods.csv
└── summary.json
```

### Selection limitation

Normal-only ablation can identify unstable scores and thresholds that generate
too many normal false alarms. It cannot determine which score detects anomalies
best. The final score and threshold must be confirmed on labeled anomalous data
using AUROC, AUPRC, recall, precision, F1 and false-positive rate. A low normal
FPR alone can result from an excessively insensitive score or threshold.
