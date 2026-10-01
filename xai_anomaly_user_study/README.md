
---
title: Explainable Anomaly Detection User Study
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
---

# Explainable Anomaly Detection User Study

A small Gradio application for evaluating whether an anomaly/explanation map
identifies an unexpected condition for the **right reason** and in the **right
safety area**.

It is designed to work:

1. locally on your machine, and
2. online as a Hugging Face Space.

## 1. Prepare the videos

The web app should use browser-friendly MP4/H.264 files.

### Generate the two ADVIS dashboard conditions

Use the study-specific inference entry point to generate two synchronized
videos from the same inference run:

```bash
python scripts/infer_offline_study.py \
  --config configs/cf_dataset_epito.yaml \
  --input_type video \
  --scenario 8_16 \
  --topic /camera/back_view/image_raw \
  --safety_areas ALL \
  --threshold_strategy percentile \
  --offset 3 \
  --sigma 1.0 \
  --quantile 0.99
```

Outputs are written under:

```text
results/V6/offline_inference/study_video/
```

The `_TL_TR.mp4` condition contains **Input View + Unexpected Situations
View**. The `_BL_BR.mp4` condition contains **AI View + Details**. Both videos
have the same frame count and FPS, allowing controlled comparison in the user
study.

OpenCV writes these intermediate files with the portable `mp4v` encoder. For
browser and Hugging Face Spaces playback, transcode the selected study clips
to H.264 before copying them into `xai_anomaly_user_study/videos/`:

```bash
ffmpeg -i INPUT.mp4 -c:v libx264 -pix_fmt yuv420p -an OUTPUT.mp4
```

For your current file:

```bash
ffmpeg \
  -i reports/explainable_ai/Unexpected_box.mov \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -c:a aac \
  xai_anomaly_user_study/videos/Unexpected_box.mp4
```

If the source has no audio, FFmpeg may warn about the audio stream. In that
case use:

```bash
ffmpeg \
  -i reports/explainable_ai/Unexpected_box.mov \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -an \
  xai_anomaly_user_study/videos/Unexpected_box.mp4
```

Put all study clips in:

```text
videos/
```

## 2. Describe each clip

Edit `videos.csv`:

```csv
video_id,filename,ground_truth,expected_area,event_type
V001,Unexpected_box.mp4,unexpected,RoboArm,unexpected_object
V002,normal_01.mp4,normal,None,normal_palletizing
V003,operator_fall_01.mp4,unexpected,RoboArm,operator_fall
```

The participant never sees the ground-truth metadata.

## 3. Run locally

Create an environment and install dependencies:

```bash
cd xai_anomaly_user_study
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the URL printed by Gradio, normally:

```text
http://127.0.0.1:7860
```

Responses are stored locally in:

```text
data/study.db
```

The researcher panel can export:

```text
data/responses_export.csv
```

## 4. Deploy to Hugging Face Spaces

Create a new **Gradio Space**, then upload/push the contents of this directory.

A basic repository will contain:

```text
app.py
requirements.txt
README.md
videos.csv
videos/
```

### Persistent response storage

A normal Hugging Face Space filesystem should not be treated as permanent
study storage. This app therefore supports synchronization to a separate
Hugging Face **Dataset repository**.

Create a private Dataset repository, for example:

```text
rashidrao/distrimuse-xai-study-responses
```

In your Space settings, create these secrets/variables:

```text
HF_TOKEN       = a Hugging Face write token
HF_DATA_REPO   = rashidrao/distrimuse-xai-study-responses
```

Do **not** put the token in the repository.

Every submitted response is first written to SQLite and then merged into
`responses.csv` in the configured Dataset repository.

For a small supervised research study this is convenient. If you later expect
many simultaneous public participants, use a transactional database such as
PostgreSQL/Supabase instead.

## 5. Participant behavior

- Each participant enters a participant ID such as `P001`.
- Clip order is deterministically randomized per participant.
- Refreshing and entering the same ID resumes the locally recorded study.
- Ground truth is hidden from participants.
- The database prevents duplicate participant/video records.
- Response time is recorded.
- Comments are optional.

## 6. Questions currently collected

1. Does the explanation map indicate an unexpected condition?
2. Does the highlighted region correspond to the actual cause of the
   unexpected condition?
3. Is the unexpected condition highlighted in the correct safety area?
4. How well does the anomaly map localize the actual cause? (1–5)
5. How confident are you in your assessment? (1–5)
6. Optional comment.

## 7. Important study-design note

For a stronger explainability experiment, consider a later two-stage version:

**Stage A:** show only the input/scene and ask the participant where they think
the unexpected condition occurs.

**Stage B:** reveal the anomaly map and ask whether the map agrees with their
interpretation.

That reduces the risk that the model's own visualization anchors the human
judgment.
