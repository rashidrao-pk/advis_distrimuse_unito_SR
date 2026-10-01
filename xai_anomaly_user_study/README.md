
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


## Participant progress and video counts

The participant interface now shows:
- total number of available video sequences,
- typical clip duration,
- approximate completion time,
- participant-specific completed count,
- participant-specific remaining count,
- a progress bar,
- current sequence number.

The participant view intentionally does **not** reveal the number of normal vs.
unexpected clips, because that could bias the answers.

The **Researcher tools** section shows:
- total number of videos,
- ground-truth distribution,
- expected safety-area distribution,
- event-type distribution,
- participants started,
- participants who completed all available videos,
- total stored responses.

Use **Refresh researcher summary** to update these counts.


## Automatic participant IDs

New participants no longer need to invent an ID manually.

Click **Start new participant** and the app generates an anonymous ID such as:

```text
P-A1B2C3
```

The ID is reserved immediately in the local SQLite `participants` table. A
participant can later enter the same ID and click **Resume participant**.

Random short IDs are preferable to sequential IDs such as `P001`, `P002` for
an online Hugging Face Space because multiple people may begin at nearly the
same time and a Space may restart.

The participant should keep the generated ID until the study is complete.


## Wide two-panel video layout

The video component is now hidden until a trial starts and does not reserve a
fixed 560-pixel-high rectangle. The browser uses the source video's intrinsic
aspect ratio, so a wide two-window video occupies only the height it actually
needs.

## Color-assisted response choices

The interface displays:

- 🟢 Yes
- 🟡 Partially
- 🔴 No
- ⚪ Cannot determine

These colors are only a visual aid. The database continues to store the clean
values `Yes`, `Partially`, `No`, and `Cannot determine`, so existing analysis
remains compatible.


## ZeroGPU-compatible researcher QA

This version includes one real GPU-backed researcher utility using
`@spaces.GPU(duration=15)`.

The **GPU video quality check**:
- opens the selected study video,
- samples the middle frame,
- transfers the frame to CUDA,
- resizes it on GPU with PyTorch,
- computes luminance and contrast on GPU,
- returns a simple visual-quality report.

This is a researcher QA utility only. It does not change participant answers,
ground-truth labels, or the questionnaire flow.

The participant-facing app remains the same Gradio study interface.


## DistriMuSe-branded participant interface

The participant-facing UI now includes:
- DistriMuSe / UC3 branding,
- Safe Interaction with Robots context,
- University of Torino and Explainable AI badges,
- links to the DistriMuSe project, ADVIS project page, and GitHub repository,
- a participant-facing reference card identifying each curated sequence as
  **NORMAL** or **UNEXPECTED / ANOMALOUS**.

Question 1 now asks what the explanation map indicates, with explicit choices:
- Normal / no unexpected condition indicated,
- Unexpected / anomalous condition indicated,
- Cannot determine.

For normal clips, the reference card tells the participant that no unexpected
condition is expected and asks them to judge whether the explanation stays
appropriately quiet. For anomalous clips, it asks them to judge whether the
highlighted region explains the unexpected condition for the right reason and
in the right safety area.


## Theme and contrast fixes

The interface now uses a complete theme-safe color system for both light and
dark display modes.

Fixes include:
- readable body text in Gradio Light, Dark, and System themes,
- consistent card backgrounds and borders,
- high-contrast study statistics,
- readable inputs, labels, radio options, and accordions,
- explicit dark-mode variables,
- a high-contrast DistriMuSe header and footer,
- responsive spacing on smaller displays.

The app also uses a neutral Gradio `Soft` theme underneath the custom
DistriMuSe styling so native components remain consistent.


## Radio-button interaction fix

Participant input components are now explicitly interactive. The custom theme
no longer applies styling to generic Gradio `.wrap label` elements, which can
collide with Gradio's internal radio-button markup in some versions.

The three radio questions now have dedicated element IDs and pointer/cursor
rules so the answer cards remain selectable in Light, Dark, and System themes.


## Radio-button compatibility fix

The Space is now pinned to **Gradio 5.49.1** in both `README.md` and
`requirements.txt`. Previously, the README requested 5.49.1 while
`requirements.txt` allowed any Gradio version below 7, which could install
Gradio 6.x and change frontend behavior.

The custom CSS no longer overrides `pointer-events` on radio-button internals.
Gradio now handles radio interaction natively.

Required-question validation is implemented server-side without JavaScript.
Missing required fields show a red `Required` message directly below the
corresponding question and a summary warning above the questionnaire.


## Video-completion gate

Participants cannot answer a trial before the video reaches the end.

For each trial:
1. the video is displayed,
2. the questionnaire and **Submit & Next** are hidden,
3. the browser detects the video's `ended` event,
4. the server marks the trial as watched,
5. the study reference and questionnaire are unlocked.

There is also a server-side `video_watched` check, so a response is rejected if
Submit is triggered without completing the video.

After playback, the study reference explicitly shows:
- whether the curated sequence is **Normal** or **Unexpected / Anomalous**,
- the reference condition/event,
- the relevant safety area for an anomalous sequence.

For normal sequences, the wording of the explanation questions changes so the
participant judges whether the anomaly map correctly avoids false highlights.


## Questionnaire and Researcher Tools pages

The interface now has two top-level pages/tabs.

### Questionnaire
Contains participant-facing content only:
- study introduction,
- anonymous participant ID,
- progress,
- required video viewing,
- normal/anomalous reference information,
- questionnaire,
- missing-answer validation.

### Researcher Tools
Contains:
- current study/dataset summary,
- participants and submitted evaluation counts,
- completion/coverage,
- participant progress table,
- map/reference agreement,
- right-reason and right-area indicators,
- localization and confidence averages,
- response distributions,
- CSV export,
- GPU Video Quality Check at the end.

The dashboard reports several descriptive indicators of explanation effectiveness
rather than inventing a single composite score.


## Quick 1–5 radio scales

Questions 4 and 5 now use compact single-line radio buttons instead of sliders.

- Q4: `1  2  3  4  5`
- Q5: `1  2  3  4  5`

The scale endpoints are shown below each question so participants can answer
with one quick click. The stored values remain numeric integers from 1 to 5,
so existing CSV exports and researcher-dashboard analysis remain compatible.


## Categorized event-type distribution

The Researcher Tools page now separates the event-type distribution into:

- **Normal workflow**
- **Unexpected / anomalous**

Each group shows the number of videos in that category and the count of each
event type. This replaces the previous single flat event list.
