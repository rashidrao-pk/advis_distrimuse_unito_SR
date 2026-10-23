# R2AD: dataset to trained and evaluated model

This is the current workflow for **V6, back_view, combined SafetyZone, extracted full-resolution images, virtual CSV splits, and the authoritative combined annotations**. Run commands from the repository root. Use this guide instead of older examples that assign every frame in scenarios 8–16 an anomaly label or require copied split directories.

## Progress tracker

Update this table after each gate below passes. Status updated on 2026-10-23 is based on this checkout and the outputs shared in chat. Epito storage and training jobs are not accessible from this local session.

| Step | Current status | Evidence required to mark complete |
|---|---|---|
| Environment on Epito | Partially verified | Preprocessing ran in `pt_312`; dependency imports and allocated CUDA device still need checking |
| Authoritative annotations | Complete locally | Combined CSV parses: 16,774 source frames; 9,452 normal, 6,583 anomalous, 739 unknown under the SafetyZone aggregation rule |
| Full extraction | Unverified on Epito | Every configured scenario has full-resolution frames and successful extraction logs; first-50 samples do not establish completion |
| SafetyZone union mask | Generated; alignment reviewed locally | Epito mask creation reported 1920×1080 and 19.02% coverage; inspect a representative Epito frame |
| Grid and random patch previews | Complete locally; review Epito inputs | Local grid covered 91.53% of mask; 48 random patches with seed 42 covered 95.54%; source frame was scenario 1_0 |
| Annotation/frame alignment | Pending verification | Several normal/anomalous transitions visually match the same source frame IDs |
| Annotated virtual split | Implementation tested; Epito generation unverified | Fresh annotated split, nonempty CSVs, source paths exist, scenario separation passes |
| One-epoch smoke training | Monitoring at step 1900 reported; completion unverified | Finite losses, checkpoint, monitoring images, final completion log |
| Full training | Pending | Chosen epoch budget completes; retained best checkpoint and run provenance |
| Threshold calibration | Pending | Normal validation scores and matching `threshold.json` |
| Labeled test evaluation | Pending | Complete scores with ground truth, coverage inspected, both classes represented |
| Metrics | Pending | `metrics.json` generated with exact file joins and fixed validation threshold |
| Shift robustness | Pending | Normal-only held-out inputs, `robustness.csv`, plots |
| Comparison with ADVIS | Pending, optional follow-up | Same held-out frames/labels and independently calibrated method thresholds |

Get a fresh read-only artifact report on the machine holding the data:

```bash
python scripts/check_r2ad_workflow.py \
  --config configs/cf_dataset_epito.yaml \
  --split-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split \
  --run-dir /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/runs/SafetyZone_run01
```

`PASS` means that particular data check passed. `PRESENT` only means a nonempty artifact exists. `PENDING` means missing; `REVIEW` requires inspection; `FAIL` identifies a failed check. This checker does not submit jobs, load checkpoints, modify data, or prove extraction/training completion. Do not mark a step complete just because a file exists. Save its output in your experiment notes and revisit the table after each stage.

## 1. Understand the repository and chosen model

| Component | Role in this experiment |
|---|---|
| `configs/cf_dataset_epito.yaml` / `cf_dataset_mac.yaml` | Machine-specific bags, masks, full-frame sources, split references and result defaults |
| `scripts/process_rosbags_to_frames_rrad.py` | ROS-free MCAP extraction, preserves source resolution and original per-topic frame indices |
| `scripts/process_rosbags_from_config.py` | Other extraction/masked-video workflow; useful annotation context, not required for the R2AD image route |
| `reports/safety_area_annotations/saved_annotation/` | Human ground truth; use only the combined CSV for this run |
| `r2ad/safetyzone.py`, `scripts/preprocess_r2ad.py` | Union masks, validate every frame, group normal recordings, produce virtual splits |
| `r2ad/annotations.py`, `frame_manifest.py` | Source-ID annotation joins and CSV reference loading |
| `r2ad/dataset.py` | Online random rectangles, matched image/mask geometric augmentation, brightness augmentation |
| `r2ad/core.py` | Encoder/decoder/discriminator, rectangle sampling, inference, residual maps and stitching |
| `scripts/train_r2ad.py` | VAE-GAN optimization, validation MSE, best checkpoint, monitoring |
| `scripts/evaluate_r2ad.py` | Fixed-grid reconstruction, normal threshold calibration, test scoring, overlays and ground truth export |
| `scripts/compare_scores.py` | Binary classification metrics from scores and explicit labels |
| `scripts/stress_test_r2ad.py` | Synthetic image translations, rotations and scale changes on normal test frames |
| `scripts/preview_r2ad_mask.py` | Mask overlay, sampled rectangle coverage, grid/random visualization |
| `r2ad/mcap_data.py`, `scripts/index_rosbags_r2ad.py` | Alternative direct-MCAP data route; not used in this runbook and not the same CSV format |
| Other ADVIS training/inference scripts, `services/`, `zenoh_dashboard/` | Existing area-crop/TAAS and deployment workflows; not dependencies of this offline R2AD experiment |
| `hpc/setup_env.sh`, `pixi.toml`, requirements files | Compute/environment setup |
| `tests/`, `doc/` | Verification and other workflow references |

One SafetyZone model learns normal appearance across the union of PLeft, PRight, RoboArm and ConvBelt. It outputs a frame score and spatial residual map; it does not currently output four independent area alarms.

Training samples source rectangles around 224×224 with size variation of 0.9–1.1, at least 85% inside the mask, then resizes each rectangle to **128×128 network input**. Six samples per frame means six independently sampled patches per epoch, not six permanent crops. Source images and masks are not resized during dataset preparation.

Training combines reconstruction MSE, KL regularization and adversarial losses. Validation uses reconstruction MSE with the latent mean. Each validation access samples a fresh random rectangle even with augmentation off; best-checkpoint selection therefore has sampling noise. Fixed monitoring probes are held in memory for visual comparison, but they do not make epoch validation deterministic.

Inference/calibration use fixed overlapping 224×224 rectangles at stride 64. They reconstruct with the latent mean, compute RGB mean absolute residual, smooth it, stitch with Hann weights, and score the covered mask pixels at spatial q99. This score differs from production TAAS and needs its own threshold. The random preview shows training sampling without the geometric/brightness augmentation; it does not switch inference to random patches.

## 2. Prepare the environment and an isolated run

On Epito, obtain a GPU allocation using the existing helper if appropriate for your account:

```bash
bash hpc/setup_env.sh
```

The helper requests an A100 through Slurm, activates `/beegfs/home/mrashid/pt_312`, and starts an interactive tmux shell. An existing `pt_312` prompt alone does not prove you are on an allocated GPU node. Alternatively use your already allocated compute session. Mac commands can use `pixi run python` and the Mac config; use the Epito Python environment for GPU training.

```bash
python -c "import sys, torch, cv2, yaml, numpy, pandas, sklearn, matplotlib, tqdm, mcap, mcap_ros2; print(sys.executable); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If imports are missing, install the relevant project requirements into the selected environment. `requirements-extra.txt` contains the image/model dependencies; `requirements-mcap.txt` contains the MCAP packages. Training also imports `tqdm`. Keep a working cluster PyTorch/CUDA installation rather than replacing it just to extract images.

Set these variables in each new shell (edit run01 to a new run name for another experiment):

```bash
export R2AD_CONFIG=configs/cf_dataset_epito.yaml
export R2AD_SPLIT=/beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split
export R2AD_RUN=/beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/runs/SafetyZone_run01
mkdir -p "$R2AD_RUN"
git rev-parse HEAD > "$R2AD_RUN/git_commit.txt"
git diff > "$R2AD_RUN/source_changes.patch"
python -m pip freeze > "$R2AD_RUN/environment.txt"
```

**Completion gate:** imports pass, CUDA is available for planned GPU commands, run directory is separate from previous experiments. A commit ID plus `git diff` does not preserve untracked source files: commit/archive them separately if your run uses any.

## 3. Extract full-resolution source frames

The configured normal root holds scenarios 1–7. The unexpected root holds scenarios 8–16 as **held-out recordings**, not an assertion that each frame is anomalous. Use the same camera throughout.

The first-50 command is a smoke test only. For a complete extraction into empty source scenario directories:

```bash
python scripts/process_rosbags_to_frames_rrad.py \
  --config "$R2AD_CONFIG" \
  --all --camera back_view --process-to frames --progress
```

Expected layout:

```text
V6/r2ad/sources/
  normal/scenario_1_0/frame_000000.png
  normal/scenario_2_0/frame_000000.png
  ...
  unexpected/scenario_8_0/frame_000000.png
  ...
```

If those directories already contain the 50-frame smoke extraction, the script rejects existing frame files. Review the location before explicitly rerunning with `--overwrite`; this rewrites existing frames and fills in the rest. It does not remove stale files from a previous extraction. A fresh `--output-dir` is an alternative, but then update `r2ad.normal_source` and `unexpected_source` to its `normal` and `unexpected` children.

Do not use `--max-frames 50` for a full experiment. `--save-every-n` retains the original frame index in the filename; only change it after confirming annotation alignment. Successful decoding/saving must not silently renumber a source frame.

**Completion gate:** every configured MCAP is found, every selected camera topic is present, final saved counts match source counts for full extraction, failures are zero, and no scenarios are omitted. Per-scenario extraction metadata is under each category's `_metadata/`; inspect logs too, since metadata alone does not prove all source frames were exported. The current extractor can still report a successful scenario if one of multiple requested topics is empty; this run uses one camera and you should verify its positive count explicitly.

## 4. Verify masks and source/annotation alignment

```bash
python scripts/preprocess_r2ad.py --config "$R2AD_CONFIG" --mask-only

python scripts/preview_r2ad_mask.py \
  --image /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/sources/normal/scenario_1_0/frame_000000.png \
  --mask /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_SafetyZone.png \
  --output-dir "$R2AD_RUN/preview_grid" --sampling grid \
  --patch-size 224 --stride 64 --min-coverage 0.85

python scripts/preview_r2ad_mask.py \
  --image /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/sources/normal/scenario_1_0/frame_000000.png \
  --mask /beegfs/home/mrashid/datasets/AD/SR/V6/masks/Mask_Generation_v6_SafetyZone.png \
  --output-dir "$R2AD_RUN/preview_random" --sampling random \
  --patch-size 224 --num-patches 48 --seed 42 --min-coverage 0.85
```

Inspect `preview.png`, full-resolution `overlay.png`, `patch_coverage.png`, and `coverage.json`. The supplied union mask covers 394,399 of 2,073,600 pixels (19.02%). Patch filtering can leave thin parts uncovered; red means uncovered mask pixels. A random run's coverage is specific to its seed/count, not a guarantee for every training sample.

For ground truth use only:

```text
reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv
```

It supersedes the individual CSVs for this run. Match `source_scenario_id`, `camera` and `source_frame_id`; cumulative `frame_id` and `frame_position` are not original bag indices. CSV Mac image paths are not used to locate Epito frames. Check multiple frames around anomaly transitions against the original annotated images/video. The parser can join equal numeric IDs; it cannot prove that two extraction pipelines used the same timing/order.

**Completion gate:** mask and image are both 1920×1080, overlay matches the intended areas, selected patch coverage is acceptable and recorded, and source-ID alignment is visually confirmed. If annotated frame indices came from downsampled/reordered frames, create an explicit mapping before proceeding.

## 5. Generate the annotation-aware virtual split

Use a fresh empty output directory. The earlier nonempty `full_frames` error is resolved by choosing this new location, without deleting old data.

```bash
python scripts/preprocess_r2ad.py \
  --config "$R2AD_CONFIG" \
  --transfer manifest --split-mode grouped \
  --val-fraction 0.20 --test-fraction 0.20 --seed 42 \
  --annotations reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  --camera back_view --output-dir "$R2AD_SPLIT"
```

The normal source is split by entire top-level scenario folders. Fractions apply to the number of groups, not exactly the number of frames. Groups are shuffled reproducibly with seed 42. At least three normal groups are required. Separate repeats sharing the same setup may still be correlated; if you need session-level isolation, reorganize the grouping or split policy before the experiment. Do not switch to temporal splitting to obtain a stronger evaluation.

All unexpected-source recordings stay in test, including confirmed normal frames. For the union mask:

| Area labels for a source frame | SafetyZone label |
|---|---|
| Any `Anomalous` | `anomalous` |
| All four `Normal` | `normal` |
| Otherwise, including missing labels / `Verify` | `unknown` |

Normal-source recordings retain the configured normal assumption; the combined CSV labels the held-out unexpected source. Missing annotations are never silently treated as normal there.

| Output | Meaning |
|---|---|
| `train.csv`, `val.csv` | Normal frames from separate normal-source scenario groups |
| `test.csv` | Known normal/anomalous test frames only |
| `normal_test.csv` | All confirmed normal test frames, including normal intervals in unexpected recordings |
| `unknown_test.csv` | Unresolved test frames, excluded from scored evaluation |
| `manifest.csv` | All split assignments, labels, source paths and per-area labels |
| `metadata.json` | Split groups/counts, seed, input roots, annotation CSV path |

No image copies or symlinks are created. CSV paths reference original images: keep them available. Preserve the CSVs rather than regenerating the split between experiments.

Optionally update the existing `r2ad` keys to these files for default commands:

```yaml
  preprocessed_dir: /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split
  train_frames: /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/train.csv
  val_frames: /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/val.csv
  test_frames: /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/test.csv
  normal_test_frames: /beegfs/home/mrashid/datasets/AD/SR/V6/r2ad/annotated_split/normal_test.csv
```

Commands below pass explicit split paths so they also work without this config edit. Leave `--dataset-source images` (default); `--manifest` on the training script refers to the different direct-MCAP route.

```bash
python scripts/check_r2ad_workflow.py --config "$R2AD_CONFIG" \
  --split-dir "$R2AD_SPLIT" --run-dir "$R2AD_RUN"
cp "$R2AD_CONFIG" "$R2AD_RUN/config.yaml"
cp "$R2AD_SPLIT/metadata.json" "$R2AD_RUN/split_metadata.json"
sha256sum "$R2AD_SPLIT"/*.csv > "$R2AD_RUN/split_checksums.txt"
sha256sum reports/safety_area_annotations/saved_annotation/scenario_8_16_back_view_annotations.csv \
  > "$R2AD_RUN/annotation_checksum.txt"
```

**Completion gate:** all data checks pass, train/validation contain only normal labels, no source recording crosses splits, test includes both classes for full metrics, unknown counts are reviewed, and the annotation provenance is the combined CSV. Hash commands above are for Linux/Epito; use `shasum -a 256` on Mac.

## 6. Run one smoke-training epoch

Before training, preview normal train/validation frames and anomalous test frames:

```bash
python scripts/preview_r2ad_splits.py \
  --config "$R2AD_CONFIG" --split-dir "$R2AD_SPLIT" \
  --output-dir "$R2AD_RUN/split_preview" --samples 6 --seed 42
```

Open `split_preview/index.html` or the `train_normal.jpg`, `val_normal.jpg`, and
`test_anomalous.jpg` contact sheets. Each sample shows the original frame beside
the SafetyZone overlay, scenario/frame IDs, and four annotation labels. Sampling
spreads across scenarios before selecting additional frames from a recording.
The script checks full CSVs for source-file and scenario overlap, incorrect split
assignments, and unexpected labels; findings and sample source paths are saved to
`report.json`. A failed check exits with status 1. Missing anomalous examples can
mean the first-50 extraction missed the anomaly; do not infer that all test frames
are anomalous. Visual review is a spot check, not proof every annotation is correct.

Use the actual split but isolate the checkpoint from the full run. This verifies execution; on a large dataset even one epoch can take time.

```bash
mkdir -p "$R2AD_RUN/smoke"
set -o pipefail
python -u scripts/train_r2ad.py \
  --config "$R2AD_CONFIG" --safety-area SafetyZone \
  --train-dir "$R2AD_SPLIT/train.csv" --val-dir "$R2AD_SPLIT/val.csv" \
  --output "$R2AD_RUN/smoke/model_r2ad.pt" \
  --epochs 1 --batch-size 8 --workers 4 --device cuda \
  --patch-size 224 --min-coverage 0.85 --samples-per-frame 6 \
  --latent-dims 64 --seed 42 --monitor-every 100 --monitor-probes 4 \
  2>&1 | tee "$R2AD_RUN/smoke/train.log"
```

The script has no `--resume` option and starts a new model each invocation. It saves best validation checkpoints, not optimizer state for resuming interrupted training. If you deliberately create a tiny smoke subset, select it from the existing train/val CSVs without mixing scenario groups and retain the original split for the full run.

**Completion gate:** final `[done] best validation MSE` appears, finite losses and validation MSE are printed, checkpoint exists, and `smoke/live_monitor/` contains probe images and loss plots. Check that reconstructions are spatially aligned and not uniformly blank; one epoch is not a quality benchmark. If CUDA runs out of memory, reduce batch size; if worker loading fails, try `--workers 0` to diagnose.

## 7. Train the full baseline

This is an explicit baseline setting, not a claim that 50 epochs is optimal. The YAML currently has larger defaults, so specify epochs/batch size rather than inheriting them accidentally.

```bash
set -o pipefail
python -u scripts/train_r2ad.py \
  --config "$R2AD_CONFIG" --safety-area SafetyZone \
  --train-dir "$R2AD_SPLIT/train.csv" --val-dir "$R2AD_SPLIT/val.csv" \
  --output "$R2AD_RUN/model_r2ad.pt" \
  --epochs 50 --batch-size 8 --workers 4 --device cuda \
  --patch-size 224 --min-coverage 0.85 --samples-per-frame 6 \
  --latent-dims 64 --lr 0.0002 --kl-weight 0.0001 --adv-weight 0.01 \
  --seed 42 --monitor-every 100 --monitor-probes 4 \
  2>&1 | tee "$R2AD_RUN/train.log"
```

Monitoring outputs are `live_monitor/batch_losses.csv`, `batch_losses.png`, original/reconstructed/error grids per monitoring step, and epoch probe grids. Each patch is now a **column**, and original, latent-mean reconstruction, and absolute error form three rows. By default the image has four fixed validation columns and two fresh random augmented training columns. Change these with `--monitor-probes 4 --monitor-random-probes 2`; set the latter to 0 to show only fixed probes. Fresh training examples change at each monitoring event, while fixed validation examples allow comparisons of identical inputs over time. Error maps keep the same 0–0.5 color range.

`live_monitor/fixed_probes.pt` preserves the fixed input tensors. Existing PNGs and an already running Python process are not retroactively updated by a source edit. Old fixed tensors were not saved by earlier versions, so an old PNG alone cannot reproduce the exact original monitoring calculation. A new training invocation starts from scratch; there is no resume option.

Training prints source-frame counts, train/validation patches and batches per epoch, total planned optimizer steps, sampling/augmentation behavior and output locations before model initialization. Batch progress includes elapsed time and estimated remaining run time. The first-epoch estimate excludes unmeasured validation/epoch overhead and is marked accordingly. Later estimates use measured epoch overhead. Validation has its own progress bar. Each completed epoch reports training, validation and total epoch times, total elapsed and estimated remaining/projected total time. These are wall-clock estimates for the observed workload, not promises about cluster scheduling or other hardware.

The checkpoint directory also contains:

| Artifact | Contents |
|---|---|
| `training_summary.json` | Resolved arguments, data counts, status, epoch history, best epoch/MSE, elapsed and estimated remaining seconds |
| `training_report.md` | Human-readable account of data, sampling, losses, monitoring and epoch timing/results |
| `epoch_metrics.csv` | Train/validation MSE, phase/epoch timing and checkpoint improvement per completed epoch |

Reports update during training even when images are disabled with `--monitor-every 0`. `status=completed` is written only after the training loop finishes normally; an interrupted run can leave a stale `training` status and earlier elapsed-time estimate. Training phase time includes batch loading, optimization and step monitoring; total epoch time also includes validation, epoch plots and checkpoint saving. At completion the final message reports total run time including setup and the best epoch.

### Learning dashboard and red error maps

`training_progress.png` beside the checkpoint has three panels, similar to the loss-progression reference:

1. Log-scale training/validation reconstruction MSE, raw KL, weighted KL, raw generator GAN loss, weighted GAN loss and generator total.
2. Discriminator accuracy and real-class F1 computed from real/reconstruction logits at threshold zero (`sigmoid=0.5`). These are training diagnostics, **not anomaly-classification metrics**. A strong discriminator is not proof the autoencoder detects anomalies; balanced adversarial training can have discriminator accuracy near 50%.
3. Generator adversarial loss and discriminator BCE to inspect adversarial balance.

The latest plot updates at monitoring intervals using sample-weighted partial-epoch statistics (validation is marked pending), and after each epoch using completed metrics. The discriminator metrics use aggregated confusion counts across the whole epoch, rather than averages of batch F1 values. All epoch values are retained in `epoch_metrics.csv` and `training_summary.json`. The plot is generated after completed epochs even with `--monitor-every 0`. No annealing curve is shown because the trainer has no annealing schedule.

Learning evidence is decreasing held-out reconstruction error together with better fixed-probe reconstructions, stable losses and later labeled anomaly evaluation. Fresh random patches help inspect generalization across content but are not directly comparable image-by-image over time. Validation patches themselves are resampled each epoch, so small MSE fluctuations need not reflect a model regression.

Both training absolute-error maps and evaluation heatmaps now use **`Reds` with a fixed 0–0.5 range** on the RGB 0–1 residual scale. White means low error; increasingly dark red means larger error. Set `--map-vmin 0 --map-vmax 0.2` on training or evaluation for greater contrast, but keep the same range when comparing runs. Values outside the selected range saturate for display only; scores and saved raw `.npy` maps are unchanged. Evaluation saves an accompanying `*_scale.json` containing the colormap and limits. It no longer normalizes the heatmap separately to each frame's upper quantile. The display maximum is not the calibrated anomaly threshold.

The checkpoint is overwritten only on improved validation MSE; its stored `epoch` can be smaller than the completed training epoch. No early-stopping rule is implemented. Loss improvement alone does not establish anomaly-detection performance.

To retain periodic model snapshots, add `--save-cp-k 1` (alias `--save_cp_k 1`) to save every completed epoch, or `--save-cp-k 5` for epochs 5, 10, 15, etc. The default `0` disables periodic snapshots. They are stored under `checkpoints/model_r2ad_epoch_0001.pt` beside the best-model file (the name follows your `--output` stem). The best-validation `model_r2ad.pt` is still maintained independently. Epoch metrics record each periodic checkpoint path. These contain model weights and epoch/config/MSE information, not optimizer/RNG state; they do not add training resume support. Periodic snapshots accumulate on disk, so choose the interval for your storage budget.

**Completion gate:** the planned epoch budget and final completion line are present, monitor outputs are reviewed, and the best checkpoint is retained with code/config/environment/split provenance. Do not tune hyperparameters on test scores.

## A100 performance configuration (before starting a new run)

The optimized trainer defaults to `--precision auto`, selecting BF16 on supported CUDA devices and FP32 elsewhere. Convolution/linear forwards use autocast; reconstruction, KL, variance exponentiation and BCE loss arithmetic remain FP32. Parameters/checkpoints remain FP32. BF16 changes numerical behavior, so retain the precision in experiment provenance and compare validation quality as well as speed. Use `--precision fp32` for a numerical reference. The implementation follows [PyTorch autocast guidance](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Image datasets now cache masks by frame dimensions per worker. `--augmentation-mode crop` (default) applies the original geometric transform to the mask, samples in that transformed coordinate system, then renders only the chosen image rectangle and applies brightness/contrast there. It retains patch count, sampling/augmentation distributions and full-resolution sources. OpenCV interpolation coordinate rounding can produce small pixel differences from `--augmentation-mode full`, the legacy comparison path. Full PNG decoding is still required for each sample; no unlimited full-frame RAM cache or copied dataset is created. The direct-MCAP dataset retains its full-frame augmentation implementation.

Workers use one OpenCV thread each. `--persistent-workers` defaults on when workers are nonzero, and `--prefetch-factor 2` prepares up to two batches per worker. They are omitted from DataLoader when `--workers 0`. Persistent train and validation workers can both remain resident after the first epoch, so measure CPU/RAM use against the job's allocation. These controls follow [PyTorch DataLoader semantics](https://docs.pytorch.org/docs/stable/data.html). Exact random sequences differ from older runs with different worker lifecycles; this is not a bitwise reproducibility mode.

Monitoring now defaults to every 1,000 steps. Batch metrics append only new rows instead of rewriting all previous rows; loss-plot rendering uses at most 2,000 displayed points while the CSV retains every recorded step. Progress still reports every 100 batches. ETA waits through up to 10 warm-up batches, excludes their startup delay from its per-batch estimate, and includes measured epoch overhead after one epoch finishes.

For a comparable one-epoch A100 benchmark, preserve the current batch size 32 and use a new output directory:

```bash
python -u scripts/train_r2ad.py \
  --config "$R2AD_CONFIG" --safety-area SafetyZone \
  --train-dir "$R2AD_SPLIT/train.csv" --val-dir "$R2AD_SPLIT/val.csv" \
  --output results/r2ad/SafetyZone_fast_b32/model_r2ad.pt \
  --patch-size 224 --samples-per-frame 6 --epochs 1 \
  --batch-size 32 --workers 8 --device cuda \
  --precision bf16 --augmentation-mode crop \
  --persistent-workers --prefetch-factor 2 --monitor-every 1000
```

Compare epoch training/validation seconds and `patches_per_second` with the prior run. `data_wait_seconds` is observed host time obtaining the next batch, not a complete GPU/CPU profiler or proof of a storage bottleneck. After this baseline, try batch size 64 with another output directory; compare throughput and validation losses without changing the learning rate automatically. Fifty epochs still sample the same total patches, but a larger batch changes the number of optimizer updates and may affect quality. Eight workers need sufficient allocated CPUs; more workers are not always faster on shared BeeGFS storage.

`--tf32` and `--cudnn-benchmark` are optional, off by default; enable explicitly for a recorded performance experiment. No A100 speedup is guaranteed from local CPU tests. Changes apply only to new Python processes, and the trainer still cannot resume an already running/finished checkpoint.

## 8. Calibrate the anomaly threshold on normal validation frames

```bash
python -u scripts/evaluate_r2ad.py \
  --config "$R2AD_CONFIG" --safety-area SafetyZone \
  --input "$R2AD_SPLIT/val.csv" --checkpoint "$R2AD_RUN/model_r2ad.pt" \
  --output-dir "$R2AD_RUN/calibration" --calibrate --threshold-percentile 99 \
  --patch-size 224 --stride 64 --min-coverage 0.85 --sigma 1 \
  --batch-size 8 --device cuda
```

This computes the 99th percentile of normal validation **frame scores**; the frame score itself is spatial q99 of the stitched residual. These are two separate quantiles. It is an empirical calibration setting, not a guaranteed 1% test false-positive rate.

This baseline uses the validation recordings for both checkpoint selection and calibration. For a strictly independent calibration experiment, reserve additional normal recording groups and pass their normal-only CSV with `--input`; the current three-way splitter does not create a fourth calibration set automatically.

**Completion gate:** `calibration/scores.csv` covers every validation frame and `threshold.json` is present. Calibration has no frame limit, uses only normal data, and uses the selected checkpoint. Preserve patch size, stride, coverage and sigma for testing. Changing checkpoint, mask or scoring settings requires recalibration; the script checks numeric scoring settings but does not enforce all artifact provenance relationships.

## 9. Evaluate the labeled test frames

First inspect a few maps in a separate directory if needed. `--max-frames 20 --save-maps` selects the first 20 CSV entries; this is a visual diagnostic and may contain only normal frames. For full scoring omit the frame limit:

```bash
python -u scripts/evaluate_r2ad.py \
  --config "$R2AD_CONFIG" --safety-area SafetyZone \
  --input "$R2AD_SPLIT/test.csv" --checkpoint "$R2AD_RUN/model_r2ad.pt" \
  --threshold-json "$R2AD_RUN/calibration/threshold.json" \
  --output-dir "$R2AD_RUN/test" \
  --patch-size 224 --stride 64 --min-coverage 0.85 --sigma 1 \
  --batch-size 8 --device cuda
```

Add `--save-maps` only if you need the full map outputs and storage permits. Scores contain source path, anomaly score, predicted decision, covered mask fraction, aggregate ground truth and four area labels. For this run use `--safety-area SafetyZone`; other areas require their own mask/model/calibration and a suitable per-area test selection. Area labels beside a union-model score do not turn it into four area-specific detectors.

**Completion gate:** score rows equal `test.csv` rows, each ground truth is normal/anomalous, both classes are represented, and map alignment/coverage is reviewed on representative normal and anomalous frames. Unknown frames are excluded rather than converted into negatives.

## 10. Compute classification metrics

Create binary labels from the ground truth already exported beside the scores. Exact absolute file paths are preserved for the join:

```bash
python - <<'PY'
import os
from pathlib import Path
import pandas as pd
run = Path(os.environ['R2AD_RUN'])
scores = pd.read_csv(run / 'test/scores.csv')
labels = scores['ground_truth'].map({'normal': 0, 'anomalous': 1})
assert len(scores) and labels.notna().all(), 'Missing/unknown ground truth'
assert not scores['file'].duplicated().any(), 'Duplicate source frames'
assert labels.nunique() == 2, 'Both classes required for the full report'
pd.DataFrame({'file': scores['file'], 'label': labels.astype(int)}).to_csv(run / 'test/labels.csv', index=False)
PY

python scripts/compare_scores.py \
  --scores "$R2AD_RUN/test/scores.csv" --labels "$R2AD_RUN/test/labels.csv" \
  --threshold-json "$R2AD_RUN/calibration/threshold.json" \
  --out "$R2AD_RUN/test/metrics.json"
```

**Completion gate:** retain confusion counts, precision, recall, F1, false-positive rate, balanced accuracy, AUROC and AUPR. Report the test population, class counts, unknown exclusions and spatial mask coverage with the metrics. Neighboring frames are correlated; frame count is not the number of independent experimental trials. Aggregate metrics can hide scenario-specific failures, so inspect the exported scenario/area labels too.

## 11. Run normal-only camera-shift stress tests

```bash
python scripts/stress_test_r2ad.py \
  --config "$R2AD_CONFIG" --safety-area SafetyZone \
  --input "$R2AD_SPLIT/normal_test.csv" \
  --checkpoint "$R2AD_RUN/model_r2ad.pt" \
  --threshold-json "$R2AD_RUN/calibration/threshold.json" \
  --output-dir "$R2AD_RUN/robustness" \
  --max-frames 30 --batch-size 8 --device cuda
```

The stress script takes patch/scoring settings from `threshold.json`. Its default first 30 normal frames are a quick diagnostic, not a representative full test. It does not interpret `--max-frames 0` as unlimited; use the actual normal-test frame count to include all frames, or prepare a documented representative normal subset CSV.

**Completion gate:** `robustness.csv` and FPR plots exist; report selected frames and baseline FPR alongside shift responses. The frame is synthetically shifted while the mask remains fixed: this tests sensitivity, not physical camera registration.

## 12. Close out the experiment

- [ ] All full-training epochs completed; checkpoint selected by validation, not test.
- [ ] Split CSVs and annotation checksum preserved; original sources remain accessible.
- [ ] Config, source revision/changes, environment, command logs and seed recorded.
- [ ] Threshold belongs to this checkpoint, mask and scoring configuration.
- [ ] Complete labeled test metrics and unknown exclusions recorded.
- [ ] Representative reconstructions, maps and uncovered-mask regions reviewed.
- [ ] Robustness inputs and limits documented.
- [ ] Progress table at the top updated with paths and completion evidence.

Only then compare with existing ADVIS on the same held-out frames/labels, with each method's own normal calibration threshold. Deployment into ROS/Zenoh is a separate follow-up after the offline results are understood.

## Troubleshooting

| Symptom | Check/action |
|---|---|
| Nonempty split output error | Choose a new output directory; do not delete source frames |
| All test frames labeled anomalous/unexpected | Rebuild with the authoritative `--annotations` file and use that new `test.csv` |
| Most labels unknown | Verify camera and original source frame IDs; check annotation coverage and sampling history |
| No anomalies in extracted test frames | Remove first-50 limit; inspect full recording annotations |
| Mask dimension mismatch | Use original camera frames and matching full-resolution mask; do not silently resize |
| No rectangles satisfy coverage | Inspect mask; test smaller patch/lower occupancy in previews, then record and recalibrate the chosen settings |
| Out of GPU memory | Lower batch size; architecture contains large fully connected layers |
| Calibration/test parameter mismatch | Use identical patch size, stride, occupancy and sigma, or recalibrate |
| Checkpoint exists but run stopped | Check final training log; best checkpoints can exist before a full run completes; no resume is implemented |
| A newer run appears to reuse old metrics | Use separate run directories and explicit checkpoint/threshold/output paths |

Related: [mask and split details](../R2AD_SAFETYZONE.md), [direct-MCAP alternative](../R2AD_MCAP_NATIVE.md), [legacy R2AD overview](../R2AD.md).
