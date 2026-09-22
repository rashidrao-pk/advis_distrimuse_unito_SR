set -euo pipefail
total=18
current=0

for ooff in 1 2 3; do
    for ss in 1.0 1.5; do
        for qq in 0.99 0.98 0.97; do
        ((current += 1))
        echo "================================================================"
        echo "Combination ${current}/${total}"
        echo "Offset=${ooff}, Sigma=${ss}, Quantile=${qq}"
        echo "================================================================"
        python scripts/calibrate_threshold.py \
        --config configs/cf_dataset_epito.yaml \
        --mode val \
        --safety_area ALL \
        --dataset_version V6 \
        --threshold_strategy percentile \
        --offset "$ooff" \
        --sigma "$ss" \
        --quantile "$qq" || exit 1
        done
    done
done