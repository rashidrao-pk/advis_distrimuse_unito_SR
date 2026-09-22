## Run Evaluation

```bash
set -euo pipefail

total=6
current=0

annotations="/Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv"

scores_dir="/Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/results/V6/offline_inference"

for ooff in 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))

      scores="${scores_dir}/video_8_16_percentile_off${ooff}_sig${ss}_q${qq}_scores.csv"

      echo "================================================================"
      echo "Combination ${current}/${total}"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "Scores: ${scores}"
      echo "================================================================"

      if [[ ! -f "$scores" ]]; then
        echo "[ERROR] Score CSV does not exist: $scores"
        exit 1
      fi

      python scripts/compare_annotations_detection.py \
        --annotations "$annotations" \
        --scores "$scores"
    done

    echo "[COMPLETED] All quantiles for offset=${ooff}, sigma=${ss}"
  done

  echo "[COMPLETED] All sigma values for offset=${ooff}"
done

echo "[COMPLETED] All ${total} threshold combinations"

```

## Run Evaluation Script

```bash
chmod +x scripts/bash/run_evaluation.sh
./scripts/bash/run_evaluation.sh
```

## Run Evaluation Comparison

```bash
python scripts/compare_evaluation_results.py \
  /Users/rashid/data/PhD/datacloud_data/repos/DistriMuSe/advis_distrimuse_unito_SR/results/V6/evaluation \
  --rank-by balanced_accuracy
```

---

```bash
# Fewest false alarms

python scripts/compare_evaluation_results.py \
 results/V6/evaluation \
 --rank-by false_positive_rate

# Fewest missed anomalies

python scripts/compare_evaluation_results.py \
 results/V6/evaluation \
 --rank-by false_negative_rate

# Highest anomaly recall

python scripts/compare_evaluation_results.py \
 results/V6/evaluation \
 --rank-by recall
```
