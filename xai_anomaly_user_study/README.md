
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


## Reduced questionnaire

The participant questionnaire now has two required questions:

1. Does the explanation map correctly indicate the condition shown in the sequence?
2. Does the highlighted region correspond to the actual cause of the condition?

Both use: **Yes / No**.

Removed:
- right safety-area question,
- localization 1–5 rating,
- confidence 1–5 rating.

Legacy database/CSV columns remain for compatibility and are empty for new responses.


## Yes/No-only response format

The two participant questions now use only:
- **Yes**
- **No**

`Partially` and `Cannot determine` have been removed from both the questionnaire
and the response key.


## Database Manager

A third top-level **Database Manager** page is available for study administration.

Features:
- active participant table,
- active response table,
- database statistics,
- CSV export,
- downloadable SQLite backups,
- archive/delete one response,
- archive/delete a participant and all responses,
- restore archived participants,
- optional removal of matching Hugging Face Dataset rows,
- full active-study reset with `DELETE ALL` confirmation.

Every destructive action first creates a timestamped SQLite backup in
`data/backups/`. Deleted local records are moved to `deleted_participants`
and `deleted_responses` archive tables rather than being irreversibly destroyed.


## TP / FP / TN / FN guide

A compact four-card guide is shown immediately above the study video:

- **TP — True Positive:** anomaly detected as anomaly
- **FP — False Positive:** normal flagged as anomaly / false alarm
- **TN — True Negative:** normal detected as normal
- **FN — False Negative:** anomaly missed / missed anomaly

The guide is informational only and does not change participant responses or
the study database.


## Fixed video order

Study videos are no longer randomized per participant.

Every participant now receives the videos in ascending `video_id` order using
natural sorting (for example: `V1`, `V2`, `V3`, ..., `V10`).


## Password-protected Database Manager

The **Database Manager** tab is locked by default.

Configure the password as a Hugging Face Space Secret:

- Name: `DB_MANAGER_PASSWORD`
- Value: your chosen researcher/admin password

In Hugging Face Spaces, open **Settings → Variables and secrets → New secret**,
add `DB_MANAGER_PASSWORD`, then restart/rebuild the Space.

The password is not hard-coded in `app.py`. The database-management controls
remain hidden until the correct password is entered.


## Focused participant page

The Questionnaire page was streamlined for a short 2–5 minute evaluation.

Removed:
- long playback-lock explanation,
- sequence number / sequence progress sentence above the video,
- participant-access instructions,
- long purpose paragraph,
- long human–robot collaboration subtitle,
- participant-view bias note,
- TP/FP/TN/FN quick interpretation guide.

The participant page now prioritizes the participant controls, video, study
reference, two Yes/No questions, optional comment, and submission.


## Ultra-focused participant flow

The participant-facing page was further simplified:

- removed **Resume participant**,
- removed the visible participant ID field,
- removed the participant ID from the progress card,
- removed the full **Study overview** card with sequence/time estimates,
- renamed the primary entry action to **Start study**.

Participant IDs are still generated and stored internally for response tracking.


## Compact progress indicator

The participant progress card is now a single compact row:

`Progress | completed / total Completed | remaining Remaining`

The redundant **Available** count was removed because it duplicated the total
already shown in the completed/total value.


## Collapsible study reference

The post-video **Study reference** is now minimized by default using a collapsible
panel. The collapsed row shows only the reference type. Expanding it reveals:

- video filename from `videos.csv`,
- reference/anomalous condition,
- relevant safety area,
- one short interpretation sentence.

This keeps the participant flow compact while preserving the reference details.


## No-comment streamlined response

The participant comment field was removed.

For both Yes/No questions:
- the selected option stays fully visible,
- it gets a strong purple outline/background,
- the other option is dimmed after selection.

The legacy `comment` database/CSV column remains for compatibility and new
responses store it as an empty string.


## Softer Yes/No selection styling

The standalone Yes/No legend and helper sentence were removed.

The answer buttons now use softer visual cues:
- **Yes** has a light green tint,
- **No** has a light red tint,
- the selected option becomes clearer,
- the unselected option dims gently rather than disappearing visually.


## Cleaner question UI v2

- removed the standalone `Yes / No` legend above the questions,
- removed the light background/pill behind each question title,
- Yes and No are neutral before selection,
- selected Yes uses a soft green highlight,
- selected No uses a soft red highlight,
- the unselected option dims strongly after selection.


## Explicit selected-vs-unselected dimming

The two Yes/No questions now behave visually as follows:

- **Yes** is softly green and **No** softly red before selection.
- Selecting **Yes** keeps Yes fully visible and strongly dims/desaturates No.
- Selecting **No** keeps No fully visible and strongly dims/desaturates Yes.

This makes the chosen response immediately obvious.


## Final Yes/No clarity fix

The colored emoji circles were removed from the answer labels because they
remained visually strong even when the option was unselected.

Answers now use plain `Yes` / `No` text. After selection, the opposite option is
reduced to 20% opacity and fully desaturated. The selected option uses a soft
green (Yes) or soft red (No) background.


## Radio-dot colors

To make the two answers easier to distinguish:

- selected **Yes** uses a green radio dot,
- selected **No** uses a red radio dot,
- unselected radio controls remain neutral gray.


## Inline selection styling fix

The Yes/No visual state is now applied by a browser-side listener using inline
`!important` styles. This prevents Gradio's default blue selected styling from
overriding the intended colors.

- selected Yes: soft green + green radio control,
- selected No: soft red + red radio control,
- opposite option: 18% opacity + grayscale.

The listener also watches for Gradio re-renders and reapplies the styles.


## Stronger selected colors

Selected answer emphasis was increased while preserving the same interaction:

- selected **Yes** now uses a stronger green background, border, text, and radio dot,
- selected **No** now uses a stronger red background, border, text, and radio dot,
- the opposite option remains strongly dimmed and desaturated.


## Theme-aware Yes/No borders

The Yes/No controls now have stronger visual boundaries:

- Light theme:
  - **Yes** uses a thick green border.
  - **No** uses a thick red border.
- Dark theme:
  - both answer cards use a clear white border for contrast.
- Radio circles also receive a visible outline.
- The selected answer keeps the stronger green/red emphasis, while the opposite
  answer remains dimmed.


## Light theme by default

The participant app now starts in **light theme by default**, even when the
visitor's operating system is configured for dark mode.

Explicit `.dark` styling remains in the stylesheet for compatibility, but the
initial study view is forced to light mode on load.


## Autoplay next video

After the participant submits a response, the next study video is loaded and
playback starts automatically. A short browser-side retry loop is used so the
play request waits until Gradio has finished replacing the video source.


## Autoplay on Start study

The first video now starts automatically after the participant clicks
**Start study**.

The same autoplay helper is also used after **Submit & Next**, so both the first
trial and subsequent trials begin playback automatically.


## Focused evaluation border

The full participant task area is now enclosed in one clear visual container:

- response status and progress,
- study reference,
- video,
- both Yes/No questions,
- Submit & Next button.

This helps participants visually separate the active evaluation task from the
rest of the page.


## Self-explaining questionnaire

The participant task now starts with a concise system specification:

- **Normal:** machine palletizing, normal operator activity, pallet replacement, etc.
- **Unexpected:** unauthorized persons, unsafe movements, faults, misplaced boxes,
  or other interference outside the normal palletizing process.

The two participant questions are:

1. **Is the system behavior shown in this video consistent with the description above?**
2. **Does the highlighted area correctly explain the system decision?**

The per-video Study Reference is no longer shown to participants, avoiding a
direct cue to the expected answer.


## Bottom navigation and surveyor help

- The `Questionnaire / Researcher Tools / Database Manager` tab navigation is
  moved to the bottom of the tab content, immediately before the common footer.
- A collapsed **What should a surveyor do?** help section was added above the
  task specification.
- Expanding **Learn how to fill this** explains:
  - watch the full video,
  - left panel = input video + safety-area detections,
  - green border = normal safety area,
  - red border = potential unexpected condition,
  - verify the red detection using the right system-output/anomaly-explanation panel.


## Restored Study Reference

The per-video **Study Reference** has been restored. It appears after the video
finishes and remains collapsed by default.

The **What should the system do?** specification is also collapsed by default,
so both contextual sections stay compact unless the participant chooses to
expand them.


## System specification order

The collapsed **What should the system do?** section now presents:

1. **Anomaly / Unexpected**
2. **Normal**
3. **False Positive** — a normal situation incorrectly flagged as anomalous (false alarm)


## Tabs physically moved before footer

The actual Gradio tab navigation is now moved in the browser DOM to immediately
before the common footer. This is more reliable than reordering the tablist
inside its original wrapper.

Final order at the bottom of the page:

`Questionnaire | Researcher Tools | Database Manager`

followed by the DistriMuSe footer.


## Tabs restored to top

The main navigation tabs are back at the top of the interface:

`Questionnaire | Researcher Tools | Database Manager`

The previous browser-side tab-reordering logic was removed.
