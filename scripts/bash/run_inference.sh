set -euo pipefail

total=6
current=0

for ooff in 1 2 3; do
  for ss in 1.0 1.5; do
    for qq in 0.99 0.98 0.97; do
      ((current += 1))
    
      echo "================================================================"
      echo "Combination ${current}/${total}"
      echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
      echo "================================================================"

      python scripts/infer_offline.py \
        --config configs/cf_dataset_epito.yaml \
        --input_type video \
        --scenario 8_16 \
        --topic /camera/back_view/image_raw \
        --safety_areas ALL \
        --threshold_strategy percentile \
        --threshold_percentile 99.0 \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" \
        --scores-only
    done

    echo "[COMPLETED] ✅ All quantiles for offset=${ooff}, sigma=${ss}"
  done

  echo "[COMPLETED] ✅ All sigma values for offset=${ooff}"
done

echo "[COMPLETED] ✅ All ${total} threshold combinations"