
import base64
import csv
import cv2
import hashlib
import html as html_lib
import io
import os
import random
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import spaces
import pandas as pd
import torch
import torch.nn.functional as F

APP_DIR = Path(__file__).resolve().parent
VIDEO_DIR = APP_DIR / "videos"
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

METADATA_CSV = APP_DIR / "videos.csv"
SQLITE_PATH = DATA_DIR / "study.db"

ASSETS_DIR = APP_DIR / "assets"
HEADER_IMAGE_PATH = ASSETS_DIR / "human_robot_header.png"


def image_data_uri(path: Path) -> str:
    """Return a local image as an embedded data URI for reliable HF Spaces rendering."""
    try:
        data = path.read_bytes()
        suffix = path.suffix.lower().lstrip(".") or "png"
        mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


HEADER_IMAGE_URI = image_data_uri(HEADER_IMAGE_PATH)

# Optional Hugging Face persistent backend.
# If HF_TOKEN and HF_DATA_REPO are configured, every response is also
# synchronized to responses.csv in a Hugging Face Dataset repository.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_DATA_REPO = os.getenv("HF_DATA_REPO", "").strip()
HF_RESPONSES_FILE = "responses.csv"

DB_LOCK = threading.Lock()
HF_LOCK = threading.Lock()

QUESTION_OPTIONS = [
    "🟢 Yes",
    "🟡 Partially",
    "🔴 No",
    "⚪ Cannot determine",
]

MAP_INDICATION_OPTIONS = [
    "🔵 Normal / no unexpected condition indicated",
    "🟠 Unexpected / anomalous condition indicated",
    "⚪ Cannot determine",
]

SCHEMA_COLUMNS = [
    "participant_id",
    "trial_index",
    "video_id",
    "source_filename",
    "ground_truth",
    "expected_area",
    "event_type",
    "map_indicates_unexpected",
    "right_reason",
    "right_area",
    "localization_score",
    "confidence",
    "comment",
    "response_seconds",
    "submitted_at_utc",
]


DISPLAY_TO_VALUE = {
    "🟢 Yes": "Yes",
    "🟡 Partially": "Partially",
    "🔴 No": "No",
    "⚪ Cannot determine": "Cannot determine",
    "🔵 Normal / no unexpected condition indicated": "Normal",
    "🟠 Unexpected / anomalous condition indicated": "Unexpected",
}


def normalize_choice(value):
    """Store clean values while showing color-assisted labels in the UI."""
    return DISPLAY_TO_VALUE.get(value, value)


def init_db():
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT NOT NULL,
                trial_index INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                source_filename TEXT,
                ground_truth TEXT,
                expected_area TEXT,
                event_type TEXT,
                map_indicates_unexpected TEXT,
                right_reason TEXT,
                right_area TEXT,
                localization_score INTEGER,
                confidence INTEGER,
                comment TEXT,
                response_seconds REAL,
                submitted_at_utc TEXT NOT NULL,
                UNIQUE(participant_id, video_id)
            )
            """
        )
        conn.commit()


def load_metadata():
    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"{METADATA_CSV} does not exist. Create it from videos.example.csv."
        )

    df = pd.read_csv(METADATA_CSV, dtype=str).fillna("")
    required = {
        "video_id", "filename", "ground_truth", "expected_area", "event_type"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in videos.csv: {sorted(missing)}")

    records = df.to_dict("records")
    for rec in records:
        path = VIDEO_DIR / rec["filename"]
        rec["_path"] = str(path)
    return records



@spaces.GPU(duration=15)
def gpu_video_quality_check(filename):
    """
    Run a lightweight GPU-backed visual-quality check for a study video.

    This is intended as a researcher QA utility for ZeroGPU Spaces:
    - read the video's middle frame,
    - transfer the frame to CUDA,
    - resize it on GPU,
    - compute brightness and contrast on GPU,
    - report whether the clip appears visually usable.

    The participant questionnaire itself remains CPU/UI-bound.
    """
    if not filename:
        return "Select a video first."

    # Only allow files from the configured study video directory.
    safe_name = Path(str(filename)).name
    video_path = (VIDEO_DIR / safe_name).resolve()
    video_root = VIDEO_DIR.resolve()

    if video_root not in video_path.parents:
        return "Invalid video path."

    if not video_path.exists():
        return f"Video not found: {safe_name}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"Could not open video: {safe_name}"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if frame_count <= 0:
        cap.release()
        return f"No readable frames found in: {safe_name}"

    middle_index = max(0, frame_count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        return f"Could not read the middle frame from: {safe_name}"

    # BGR -> RGB, then real GPU processing.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device="cuda", dtype=torch.float32) / 255.0

    # Resize on GPU to a consistent QA resolution.
    tensor = F.interpolate(
        tensor,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )

    # Luminance on GPU.
    r = tensor[:, 0:1]
    g = tensor[:, 1:2]
    b = tensor[:, 2:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    mean_brightness = float(luminance.mean().item())
    contrast = float(luminance.std().item())

    duration_s = (frame_count / fps) if fps > 0 else 0.0

    # Conservative QA heuristics; this is not part of the research labels.
    if mean_brightness < 0.04:
        verdict = "⚠️ Very dark"
    elif mean_brightness > 0.96:
        verdict = "⚠️ Very bright"
    elif contrast < 0.025:
        verdict = "⚠️ Very low contrast"
    else:
        verdict = "✅ Visual QA passed"

    return (
        f"### GPU video QA — `{safe_name}`\n\n"
        f"- **Result:** {verdict}\n"
        f"- **Frames:** {frame_count}\n"
        f"- **FPS:** {fps:.2f}\n"
        f"- **Duration:** {duration_s:.2f} s\n"
        f"- **Mean luminance:** {mean_brightness:.4f}\n"
        f"- **Luminance contrast:** {contrast:.4f}\n\n"
        "The middle frame was resized and analyzed on CUDA. "
        "This QA result is for researcher validation only and is not saved "
        "as a participant response."
    )

def participant_seed(participant_id: str) -> int:
    # Deterministic randomization: same participant gets the same order
    # after refreshing/reopening the study.
    digest = hashlib.sha256(participant_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def randomized_trials(participant_id: str, metadata):
    rows = list(metadata)
    rng = random.Random(participant_seed(participant_id))
    rng.shuffle(rows)
    return rows



def participant_exists(participant_id: str) -> bool:
    with sqlite3.connect(SQLITE_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM participants WHERE participant_id = ? LIMIT 1",
            (participant_id,),
        ).fetchone()
    return row is not None


def reserve_participant_id(participant_id: str):
    now = datetime.now(timezone.utc).isoformat()
    with DB_LOCK, sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO participants (participant_id, created_at_utc)
            VALUES (?, ?)
            """,
            (participant_id, now),
        )
        conn.commit()


def generate_participant_id() -> str:
    """Generate and immediately reserve an anonymous participant ID."""
    while True:
        candidate = f"P-{secrets.token_hex(3).upper()}"
        if not participant_exists(candidate):
            reserve_participant_id(candidate)
            return candidate


def completed_video_ids(participant_id: str):
    with sqlite3.connect(SQLITE_PATH) as conn:
        rows = conn.execute(
            "SELECT video_id FROM responses WHERE participant_id = ?",
            (participant_id,),
        ).fetchall()
    return {r[0] for r in rows}


def local_response_dataframe():
    with sqlite3.connect(SQLITE_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT participant_id, trial_index, video_id, source_filename,
                   ground_truth, expected_area, event_type,
                   map_indicates_unexpected, right_reason, right_area,
                   localization_score, confidence, comment,
                   response_seconds, submitted_at_utc
            FROM responses
            ORDER BY submitted_at_utc
            """,
            conn,
        )


def sync_responses_to_hf():
    if not (HF_TOKEN and HF_DATA_REPO):
        return

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)
    local_df = local_response_dataframe()

    with HF_LOCK:
        try:
            remote_path = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=HF_RESPONSES_FILE,
                token=HF_TOKEN,
            )
            remote_df = pd.read_csv(remote_path, dtype=str).fillna("")
        except Exception:
            remote_df = pd.DataFrame(columns=SCHEMA_COLUMNS)

        # Merge remote + local. Participant/video is the study's unique response key.
        combined = pd.concat([remote_df, local_df], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(
                subset=["participant_id", "video_id"], keep="last"
            )
        combined = combined.reindex(columns=SCHEMA_COLUMNS)

        content = combined.to_csv(index=False).encode("utf-8")
        api.upload_file(
            path_or_fileobj=io.BytesIO(content),
            path_in_repo=HF_RESPONSES_FILE,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message="Update XAI user-study responses",
        )


def insert_response(payload):
    with DB_LOCK, sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            """
            INSERT INTO responses (
                participant_id, trial_index, video_id, source_filename,
                ground_truth, expected_area, event_type,
                map_indicates_unexpected, right_reason, right_area,
                localization_score, confidence, comment,
                response_seconds, submitted_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(participant_id, video_id) DO UPDATE SET
                trial_index = excluded.trial_index,
                source_filename = excluded.source_filename,
                ground_truth = excluded.ground_truth,
                expected_area = excluded.expected_area,
                event_type = excluded.event_type,
                map_indicates_unexpected = excluded.map_indicates_unexpected,
                right_reason = excluded.right_reason,
                right_area = excluded.right_area,
                localization_score = excluded.localization_score,
                confidence = excluded.confidence,
                comment = excluded.comment,
                response_seconds = excluded.response_seconds,
                submitted_at_utc = excluded.submitted_at_utc
            """,
            tuple(payload[c] for c in SCHEMA_COLUMNS),
        )
        conn.commit()

    try:
        sync_responses_to_hf()
        return True, ""
    except Exception as exc:
        # Local data is still safe. The UI informs the researcher about sync failure.
        return False, str(exc)



def estimate_study_minutes(total_videos: int, seconds_per_trial: int = 30) -> int:
    if total_videos <= 0:
        return 0
    return max(1, round((total_videos * seconds_per_trial) / 60))


def study_overview_html(metadata):
    total = len(metadata)
    estimated_minutes = estimate_study_minutes(total)
    return f"""
    <div class="study-card">
        <div class="study-card-title">Study overview</div>
        <div class="study-grid">
            <div class="study-stat">
                <div class="value">{total}</div>
                <div class="label">Total video sequences</div>
            </div>
            <div class="study-stat">
                <div class="value">~5-10 Seconds</div>
                <div class="label">Typical clip length</div>
            </div>
            <div class="study-stat">
                <div class="value">~{estimated_minutes} min</div>
                <div class="label">Estimated completion time</div>
            </div>
        </div>
        <div class="study-meta">
            The participant view intentionally does not reveal how many clips are
            normal versus unexpected, to reduce response bias.
        </div>
    </div>
    """


def participant_progress_html(participant_id: str, metadata):
    total = len(metadata)
    completed = len(completed_video_ids(participant_id))
    remaining = max(0, total - completed)
    percent = 0 if total == 0 else round((completed / total) * 100)

    return f"""
    <div class="study-card">
        <div class="progress-head">
            <div>
                <div class="study-card-title">Participant progress</div>
                <div class="study-meta">Participant ID: {participant_id}</div>
            </div>
            <div class="progress-count">{completed} / {total}</div>
        </div>

        <div class="progress-shell">
            <div class="progress-fill" style="width:{percent}%"></div>
        </div>

        <div class="study-grid">
            <div class="study-stat">
                <div class="value">{total}</div>
                <div class="label">Available</div>
            </div>
            <div class="study-stat">
                <div class="value">{completed}</div>
                <div class="label">Completed</div>
            </div>
            <div class="study-stat">
                <div class="value">{remaining}</div>
                <div class="label">Remaining</div>
            </div>
        </div>
    </div>
    """


def researcher_summary_html():
    metadata = load_metadata()
    total_videos = len(metadata)

    ground_truth_counts = {}
    area_counts = {}
    normal_event_counts = {}
    unexpected_event_counts = {}
    other_event_counts = {}

    for row in metadata:
        gt = (row.get("ground_truth", "") or "Unspecified").strip()
        area = (row.get("expected_area", "") or "None").strip()
        event = (row.get("event_type", "") or "Unspecified").strip()

        ground_truth_counts[gt] = ground_truth_counts.get(gt, 0) + 1
        area_counts[area] = area_counts.get(area, 0) + 1

        gt_key = gt.lower()
        if gt_key == "normal":
            normal_event_counts[event] = normal_event_counts.get(event, 0) + 1
        elif gt_key in {"unexpected", "anomalous", "anomaly", "abnormal"}:
            unexpected_event_counts[event] = unexpected_event_counts.get(event, 0) + 1
        else:
            other_event_counts[event] = other_event_counts.get(event, 0) + 1

    with sqlite3.connect(SQLITE_PATH) as conn:
        total_responses = conn.execute(
            "SELECT COUNT(*) FROM responses"
        ).fetchone()[0]

        participants_started = conn.execute(
            "SELECT COUNT(*) FROM participants"
        ).fetchone()[0]

        completed_participants = 0
        if total_videos > 0:
            completed_participants = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT participant_id, COUNT(DISTINCT video_id) AS n
                    FROM responses
                    GROUP BY participant_id
                    HAVING n >= ?
                )
                """,
                (total_videos,),
            ).fetchone()[0]

    def pretty_event_name(value):
        value = str(value).strip()
        if not value:
            return "Unspecified"
        return value.replace("_", " ").replace("-", " ").title()

    def pairs_to_html(values):
        if not values:
            return "<div>No data</div>"
        return "".join(
            f"<div><strong>{html_lib.escape(str(k))}</strong>: {v}</div>"
            for k, v in sorted(values.items(), key=lambda x: str(x[0]).lower())
        )

    def event_group_html(title, values, card_class, icon):
        total = sum(values.values())
        if values:
            rows = "".join(
                f"""
                <div class="event-item">
                    <span>{html_lib.escape(pretty_event_name(event))}</span>
                    <strong>{count}</strong>
                </div>
                """
                for event, count in sorted(
                    values.items(), key=lambda x: pretty_event_name(x[0]).lower()
                )
            )
        else:
            rows = '<div class="study-meta">No events in this category.</div>'

        return f"""
        <div class="event-category-card {card_class}">
            <div class="event-category-heading">
                <span>{icon} {title}</span>
                <span class="event-total">{total} video{"s" if total != 1 else ""}</span>
            </div>
            {rows}
        </div>
        """

    normal_total = sum(normal_event_counts.values())
    unexpected_total = sum(unexpected_event_counts.values())

    other_group = ""
    if other_event_counts:
        other_group = event_group_html(
            "Other / unspecified",
            other_event_counts,
            "event-other-card",
            "•",
        )

    return f"""
    <div class="study-card">
        <div class="study-card-title">Researcher summary</div>

        <div class="study-grid">
            <div class="study-stat">
                <div class="value">{total_videos}</div>
                <div class="label">Total videos</div>
            </div>
            <div class="study-stat">
                <div class="value">{participants_started}</div>
                <div class="label">Participants started</div>
            </div>
            <div class="study-stat">
                <div class="value">{completed_participants}</div>
                <div class="label">Participants completed</div>
            </div>
        </div>

        <div class="study-grid">
            <div class="study-stat">
                <div class="value">{total_responses}</div>
                <div class="label">Total responses</div>
            </div>
            <div class="study-stat">
                <div class="value">{normal_total}</div>
                <div class="label">Normal videos</div>
            </div>
            <div class="study-stat">
                <div class="value">{unexpected_total}</div>
                <div class="label">Unexpected videos</div>
            </div>
        </div>

        <div class="study-grid">
            <div class="study-stat">
                <div style="text-align:left">{pairs_to_html(ground_truth_counts)}</div>
                <div class="label" style="margin-top:8px;">Ground-truth distribution</div>
            </div>
            <div class="study-stat">
                <div style="text-align:left">{pairs_to_html(area_counts)}</div>
                <div class="label" style="margin-top:8px;">Expected-area distribution</div>
            </div>
        </div>

        <div class="event-distribution">
            <div class="study-card-title">Event-type distribution</div>
            <div class="study-meta" style="margin-bottom:10px;">
                Events are grouped by the curated sequence category.
            </div>

            <div class="event-category-grid">
                {event_group_html(
                    "Normal workflow",
                    normal_event_counts,
                    "event-normal-card",
                    "✓",
                )}
                {event_group_html(
                    "Unexpected / anomalous",
                    unexpected_event_counts,
                    "event-unexpected-card",
                    "!",
                )}
                {other_group}
            </div>
        </div>
    </div>
    """


def make_trial_state(participant_id):
    metadata = load_metadata()
    ordered = randomized_trials(participant_id, metadata)
    done = completed_video_ids(participant_id)
    remaining = [r for r in ordered if r["video_id"] not in done]
    return {
        "participant_id": participant_id,
        "ordered": ordered,
        "remaining": remaining,
        "started_at": datetime.now(timezone.utc).timestamp(),
    }


def safe_participant_id(value):
    value = (value or "").strip()
    if not value:
        return None
    if len(value) > 80:
        return None
    return value



def format_event_type(value):
    value = (value or "").strip()
    if not value:
        return "Not specified"
    return value.replace("_", " ").replace("-", " ").title()


def format_area_name(value):
    value = (value or "").strip()
    if not value or value.lower() in {"none", "n/a", "na"}:
        return "No anomalous safety area expected"

    area_labels = {
        "pleft": "Pallet Left (PLeft)",
        "pright": "Pallet Right (PRight)",
        "roboarm": "Robot Arm (RoboArm)",
        "convbelt": "Conveyor Belt (ConvBelt)",
        "conveyor": "Conveyor Belt (ConvBelt)",
        "multiple": "Multiple safety areas",
    }
    return area_labels.get(value.lower(), value)


def prewatch_reference_html():
    return """
    <div class="sequence-reference reference-locked">
        <div class="reference-kicker">VIDEO REQUIRED</div>
        <div class="reference-title">🔒 Watch the complete video to unlock the questionnaire</div>
        <div class="reference-text">
            The questions remain locked until playback reaches the end.
            After the video finishes, the study reference will show whether
            the sequence is normal or anomalous and, when applicable, the
            expected anomalous condition and relevant safety area.
        </div>
    </div>
    """


def sequence_reference_html(current):
    """
    Show the curated study reference only after the participant has watched
    the complete clip. This helps the participant evaluate the explanation
    map against the intended event and safety area.
    """
    gt = (current.get("ground_truth", "") or "").strip().lower()
    event_type = html_lib.escape(format_event_type(current.get("event_type", "")))
    area = html_lib.escape(format_area_name(current.get("expected_area", "")))

    if gt in {"normal", "expected", "no_anomaly", "no anomaly"}:
        return f"""
        <div class="sequence-reference reference-normal">
            <div class="reference-kicker">STUDY REFERENCE</div>
            <div class="reference-title">✓ NORMAL WORKFLOW</div>
            <div class="reference-text">
                No unexpected condition is expected in this sequence.
                The explanation map should remain appropriately quiet and
                should not falsely highlight a safety area.
            </div>
            <div class="reference-details">
                <span><strong>Reference condition:</strong> {event_type}</span>
                <span><strong>Expected anomalous area:</strong> None</span>
            </div>
        </div>
        """

    if gt in {"unexpected", "anomalous", "anomaly", "abnormal"}:
        return f"""
        <div class="sequence-reference reference-anomaly">
            <div class="reference-kicker">STUDY REFERENCE</div>
            <div class="reference-title">! UNEXPECTED / ANOMALOUS</div>
            <div class="reference-text">
                An unexpected condition is present. Use the information below
                as the study reference when judging whether the anomaly map
                explains the event for the right reason and in the right area.
            </div>
            <div class="reference-details">
                <span><strong>Anomalous condition:</strong> {event_type}</span>
                <span><strong>Relevant safety area:</strong> {area}</span>
            </div>
        </div>
        """

    return f"""
    <div class="sequence-reference reference-unknown">
        <div class="reference-kicker">STUDY REFERENCE</div>
        <div class="reference-title">? REFERENCE NOT SPECIFIED</div>
        <div class="reference-text">
            The sequence metadata does not specify whether this clip is normal
            or anomalous. Judge the explanation map from the visible scene.
        </div>
        <div class="reference-details">
            <span><strong>Condition:</strong> {event_type}</span>
            <span><strong>Area:</strong> {area}</span>
        </div>
    </div>
    """


def question_labels_for_current(current):
    gt = (current.get("ground_truth", "") or "").strip().lower()
    is_normal = gt in {"normal", "expected", "no_anomaly", "no anomaly"}

    if is_normal:
        return (
            "2. Does the explanation map correctly avoid highlighting a false anomalous cause?",
            "3. Does the explanation map correctly avoid falsely flagging a safety area?",
            "4. How appropriate is the spatial explanation for this normal sequence? "
            "(1 = very misleading, 5 = very appropriate)",
        )

    return (
        "2. Does the highlighted region correspond to the actual cause of the unexpected condition?",
        "3. Is the unexpected condition highlighted in the correct safety area?",
        "4. How well does the anomaly map localize the actual cause? "
        "(1 = very poor, 5 = excellent)",
    )


def unlock_questions_after_video(state):
    """
    Called only when the browser reports that the current video reached its end.
    The server marks the current trial as watched and unlocks the questionnaire.
    """
    if not state or not state.get("remaining"):
        return (
            state,
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    state["video_watched"] = True
    current = state["remaining"][0]
    q2_label, q3_label, q4_label = question_labels_for_current(current)

    return (
        state,
        sequence_reference_html(current),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(label=q2_label),
        gr.update(label=q3_label),
        gr.update(label=q4_label),
    )


def trial_display(state):
    if not state or not state.get("remaining"):
        return (
            gr.update(value=None, visible=False),
            "### Study complete\nThank you. All available clips have been rated.",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    current = state["remaining"][0]
    total = len(state["ordered"])
    completed = total - len(state["remaining"])
    trial_number = completed + 1

    video_path = current["_path"]

    if not Path(video_path).exists():
        status = (
            f"### Sequence {trial_number} / {total}\n"
            f"⚠️ Video file missing: `{current['filename']}`"
        )
        video_update = gr.update(value=None, visible=False)
        reference_html = """
        <div class="sequence-reference reference-unknown">
            <div class="reference-title">Video unavailable</div>
            <div class="reference-text">
                This trial cannot be answered because the video file is missing.
            </div>
        </div>
        """
    else:
        status = (
            f"### Sequence {trial_number} / {total}\n"
            "Watch the complete clip. The questionnaire unlocks automatically "
            "when the video finishes."
        )
        video_update = gr.update(value=video_path, visible=True)
        reference_html = prewatch_reference_html()

    # Questions and Submit stay hidden until the browser reports video completion.
    return (
        video_update,
        status,
        reference_html,
        gr.update(visible=False),
        gr.update(visible=False),
    )



def start_study_for_participant(pid):
    metadata = load_metadata()
    reserve_participant_id(pid)
    state = make_trial_state(pid)
    state["video_watched"] = False
    video, status, reference_html, q_vis, submit_vis = trial_display(state)
    progress_html = participant_progress_html(pid, metadata)

    if not state["remaining"]:
        return (
            pid,
            state,
            None,
            status,
            progress_html,
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    return (
        pid,
        state,
        video,
        status,
        progress_html,
        reference_html,
        q_vis,
        submit_vis,
    )


def start_new_participant():
    try:
        pid = generate_participant_id()
        return start_study_for_participant(pid)
    except Exception as exc:
        return (
            "",
            None,
            None,
            f"Could not start study: {exc}",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )


def resume_study(participant_id):
    pid = safe_participant_id(participant_id)
    if not pid:
        return (
            "",
            None,
            None,
            "Please enter your participant ID to resume.",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    try:
        if not participant_exists(pid):
            return (
                pid,
                None,
                None,
                "Participant ID not found. Use **Start new participant** to create a new ID.",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )
        return start_study_for_participant(pid)
    except Exception as exc:
        return (
            pid,
            None,
            None,
            f"Could not resume study: {exc}",
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )



def required_message(show):
    if not show:
        return ""
    return '<div class="field-required">⚠ Required — please answer this question.</div>'


def validation_summary(missing_labels):
    if not missing_labels:
        return ""
    items = "".join(f"<li>{label}</li>" for label in missing_labels)
    return f"""
    <div class="validation-alert">
        <div class="validation-title">Please complete the missing question(s)</div>
        <ul>{items}</ul>
    </div>
    """

def submit_response(
    state,
    map_indicates_unexpected,
    right_reason,
    right_area,
    localization_score,
    confidence,
    comment,
):
    if not state or not state.get("remaining"):
        return (
            state,
            gr.update(value=None, visible=False),
            "No active trial.",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            None,
            None,
            None,
            None,
            None,
            "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    if not state.get("video_watched", False):
        return (
            state,
            gr.update(value=state["remaining"][0]["_path"], visible=True),
            "Please watch the complete video before answering the questionnaire.",
            participant_progress_html(state["participant_id"], load_metadata()),
            prewatch_reference_html(),
            """
            <div class="validation-alert">
                <div class="validation-title">🔒 Questionnaire locked</div>
                <div>Please watch the complete video first.</div>
            </div>
            """,
            "",
            "",
            "",
            "",
            "",
            map_indicates_unexpected,
            right_reason,
            right_area,
            localization_score,
            confidence,
            comment,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    fields = [
        ("Question 1 — map indication", map_indicates_unexpected),
        ("Question 2 — right reason", right_reason),
        ("Question 3 — right safety area", right_area),
        ("Question 4 — localization score", localization_score),
        ("Question 5 — confidence score", confidence),
    ]

    missing_flags = [value in (None, "", []) for _, value in fields]
    missing_labels = [
        label for (label, _), is_missing in zip(fields, missing_flags) if is_missing
    ]

    if missing_labels:
        progress_html = participant_progress_html(
            state["participant_id"], load_metadata()
        )
        return (
            state,
            gr.update(value=state["remaining"][0]["_path"], visible=True),
            "Please complete all required questions before continuing.",
            progress_html,
            sequence_reference_html(state["remaining"][0]),
            validation_summary(missing_labels),
            required_message(missing_flags[0]),
            required_message(missing_flags[1]),
            required_message(missing_flags[2]),
            required_message(missing_flags[3]),
            required_message(missing_flags[4]),
            map_indicates_unexpected,
            right_reason,
            right_area,
            localization_score,
            confidence,
            comment,
            gr.update(visible=True),
            gr.update(visible=True),
        )

    current = state["remaining"][0]
    total = len(state["ordered"])
    trial_index = total - len(state["remaining"]) + 1
    now = datetime.now(timezone.utc)

    started_at = state.get("started_at", now.timestamp())
    response_seconds = max(0.0, now.timestamp() - started_at)

    payload = {
        "participant_id": state["participant_id"],
        "trial_index": trial_index,
        "video_id": current["video_id"],
        "source_filename": current["filename"],
        "ground_truth": current["ground_truth"],
        "expected_area": current["expected_area"],
        "event_type": current["event_type"],
        "map_indicates_unexpected": normalize_choice(map_indicates_unexpected),
        "right_reason": normalize_choice(right_reason),
        "right_area": normalize_choice(right_area),
        "localization_score": int(localization_score),
        "confidence": int(confidence),
        "comment": (comment or "").strip(),
        "response_seconds": round(response_seconds, 3),
        "submitted_at_utc": now.isoformat(),
    }

    synced, sync_error = insert_response(payload)

    state["remaining"] = state["remaining"][1:]
    state["started_at"] = datetime.now(timezone.utc).timestamp()
    state["video_watched"] = False

    video, status, reference_html, q_vis, submit_vis = trial_display(state)
    progress_html = participant_progress_html(
        state["participant_id"], load_metadata()
    )

    if synced and HF_TOKEN and HF_DATA_REPO:
        sync_note = "\n\n✅ Response saved locally and synchronized."
    elif HF_TOKEN and HF_DATA_REPO:
        sync_note = (
            "\n\n⚠️ Response saved locally, but remote synchronization failed: "
            f"`{sync_error}`"
        )
    else:
        sync_note = "\n\n✅ Response saved."

    status += sync_note

    return (
        state,
        video,
        status,
        progress_html,
        reference_html,
        "",
        "",
        "",
        "",
        "",
        "",
        None,
        None,
        None,
        None,
        None,
        "",
        q_vis,
        submit_vis,
    )


def _safe_numeric_mean(series):
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else 0.0


def _percent_of(series, values):
    if series is None or len(series) == 0:
        return 0.0
    allowed = set(values if isinstance(values, (list, tuple, set)) else [values])
    return float(series.astype(str).isin(allowed).mean() * 100)


def researcher_response_summary():
    """Descriptive completion and explanation-quality indicators."""
    metadata = load_metadata()
    total_videos = len(metadata)
    df = local_response_dataframe()

    if df.empty:
        return """
        <div class="study-card">
            <div class="study-card-title">Current response summary</div>
            <div class="study-meta">No questionnaire responses have been collected yet.</div>
        </div>
        """

    total_responses = len(df)
    participants = int(df["participant_id"].nunique())

    completed_per_participant = df.groupby("participant_id")["video_id"].nunique()
    completed_participants = int((completed_per_participant >= total_videos).sum()) if total_videos else 0
    possible = participants * total_videos
    coverage = (total_responses / possible * 100) if possible else 0.0

    q1 = df[
        df["map_indicates_unexpected"].isin(["Normal", "Unexpected"])
        & df["ground_truth"].astype(str).str.lower().isin(["normal", "unexpected"])
    ].copy()

    if len(q1):
        gt = q1["ground_truth"].astype(str).str.lower()
        pred = q1["map_indicates_unexpected"].astype(str)
        agreement = (
            ((gt == "normal") & (pred == "Normal"))
            | ((gt == "unexpected") & (pred == "Unexpected"))
        )
        map_reference_agreement = float(agreement.mean() * 100)
    else:
        map_reference_agreement = 0.0

    anomalous = df[df["ground_truth"].astype(str).str.lower() == "unexpected"].copy()
    normal = df[df["ground_truth"].astype(str).str.lower() == "normal"].copy()

    right_reason_yes = _percent_of(anomalous["right_reason"], "Yes") if len(anomalous) else 0.0
    right_reason_positive = _percent_of(anomalous["right_reason"], ["Yes", "Partially"]) if len(anomalous) else 0.0
    right_area_yes = _percent_of(anomalous["right_area"], "Yes") if len(anomalous) else 0.0
    right_area_positive = _percent_of(anomalous["right_area"], ["Yes", "Partially"]) if len(anomalous) else 0.0

    normal_reason_yes = _percent_of(normal["right_reason"], "Yes") if len(normal) else 0.0
    normal_area_yes = _percent_of(normal["right_area"], "Yes") if len(normal) else 0.0

    avg_localization = _safe_numeric_mean(df["localization_score"])
    avg_confidence = _safe_numeric_mean(df["confidence"])

    q_counts = {
        "Q1 — Map indication": int(df["map_indicates_unexpected"].notna().sum()),
        "Q2 — Right reason / false-cause avoidance": int(df["right_reason"].notna().sum()),
        "Q3 — Right area / false-area avoidance": int(df["right_area"].notna().sum()),
        "Q4 — Localization / spatial appropriateness": int(df["localization_score"].notna().sum()),
        "Q5 — Confidence": int(df["confidence"].notna().sum()),
    }

    q_rows = "".join(
        f'<div class="summary-row"><span>{html_lib.escape(k)}</span><strong>{v}</strong></div>'
        for k, v in q_counts.items()
    )

    return f"""
    <div class="study-card">
        <div class="study-card-title">Current questionnaire summary</div>

        <div class="study-grid">
            <div class="study-stat"><div class="value">{participants}</div><div class="label">Participants with responses</div></div>
            <div class="study-stat"><div class="value">{total_responses}</div><div class="label">Submitted video evaluations</div></div>
            <div class="study-stat"><div class="value">{completed_participants}</div><div class="label">Participants completed</div></div>
        </div>

        <div class="study-grid">
            <div class="study-stat"><div class="value">{coverage:.1f}%</div><div class="label">Questionnaire coverage</div></div>
            <div class="study-stat"><div class="value">{map_reference_agreement:.1f}%</div><div class="label">Map/reference agreement</div></div>
            <div class="study-stat"><div class="value">{avg_localization:.2f}/5</div><div class="label">Mean localization rating</div></div>
        </div>

        <div class="study-grid">
            <div class="study-stat"><div class="value">{right_reason_yes:.1f}%</div><div class="label">Anomalous: right reason = Yes</div></div>
            <div class="study-stat"><div class="value">{right_area_yes:.1f}%</div><div class="label">Anomalous: right area = Yes</div></div>
            <div class="study-stat"><div class="value">{avg_confidence:.2f}/5</div><div class="label">Mean confidence</div></div>
        </div>

        <div class="study-grid">
            <div class="study-stat"><div class="value">{right_reason_positive:.1f}%</div><div class="label">Reason = Yes/Partially</div></div>
            <div class="study-stat"><div class="value">{right_area_positive:.1f}%</div><div class="label">Area = Yes/Partially</div></div>
            <div class="study-stat"><div class="value">{normal_reason_yes:.1f}% / {normal_area_yes:.1f}%</div><div class="label">Normal: false-cause / false-area avoidance</div></div>
        </div>

        <div class="summary-list">
            <div class="summary-list-title">Filled-answer counts</div>
            {q_rows}
        </div>

        <div class="study-meta" style="margin-top:12px;">
            <strong>Effectiveness:</strong> these are descriptive indicators rather than
            one arbitrary composite score. Higher map/reference agreement,
            right-reason/right-area agreement, localization ratings, and normal-sequence
            false-highlight avoidance provide stronger evidence that users consider the
            anomaly explanations appropriate and understandable.
        </div>
    </div>
    """


def response_breakdown_html():
    df = local_response_dataframe()
    if df.empty:
        return """
        <div class="study-card">
            <div class="study-card-title">Response distributions</div>
            <div class="study-meta">No response data available yet.</div>
        </div>
        """

    def distribution(column, subset):
        if subset.empty:
            return '<div class="study-meta">No data available.</div>'
        counts = subset[column].fillna("Missing").astype(str).value_counts().to_dict()
        total = sum(counts.values()) or 1
        parts = []
        for key, count in counts.items():
            pct = count / total * 100
            parts.append(
                f"""
                <div class="distribution-row">
                    <span>{html_lib.escape(str(key))}</span>
                    <div class="distribution-bar-shell">
                        <div class="distribution-bar" style="width:{pct:.1f}%"></div>
                    </div>
                    <strong>{count} ({pct:.1f}%)</strong>
                </div>
                """
            )
        return "".join(parts)

    anomalous = df[df["ground_truth"].astype(str).str.lower() == "unexpected"]

    return f"""
    <div class="study-card">
        <div class="study-card-title">Response distributions</div>

        <div class="distribution-section">
            <div class="summary-list-title">Q1 — What does the map indicate?</div>
            {distribution("map_indicates_unexpected", df)}
        </div>

        <div class="distribution-section">
            <div class="summary-list-title">Q2 — Right reason (anomalous clips)</div>
            {distribution("right_reason", anomalous)}
        </div>

        <div class="distribution-section">
            <div class="summary-list-title">Q3 — Right area (anomalous clips)</div>
            {distribution("right_area", anomalous)}
        </div>
    </div>
    """


def participant_progress_table():
    metadata = load_metadata()
    total_videos = len(metadata)
    df = local_response_dataframe()

    columns = ["Participant", "Completed", "Remaining", "Progress (%)", "Last response (UTC)"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for participant_id, group in df.groupby("participant_id"):
        completed = int(group["video_id"].nunique())
        remaining = max(0, total_videos - completed)
        progress = (completed / total_videos * 100) if total_videos else 0.0

        times = pd.to_datetime(group["submitted_at_utc"], errors="coerce", utc=True)
        last_response = times.max().strftime("%Y-%m-%d %H:%M:%S") if times.notna().any() else ""

        rows.append({
            "Participant": participant_id,
            "Completed": completed,
            "Remaining": remaining,
            "Progress (%)": round(progress, 1),
            "Last response (UTC)": last_response,
        })

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Progress (%)", "Participant"], ascending=[False, True]
    ).reset_index(drop=True)


def refresh_researcher_dashboard():
    return (
        researcher_summary_html(),
        researcher_response_summary(),
        response_breakdown_html(),
        participant_progress_table(),
    )

def export_local_csv():
    df = local_response_dataframe()
    path = DATA_DIR / "responses_export.csv"
    df.to_csv(path, index=False)
    return str(path)


init_db()

CSS = """
/* =========================================================
   DistriMuSe XAI Study — theme-safe UI
   Works in Gradio Light / Dark / System display modes.
   ========================================================= */

:root {
    --dm-navy: #0B2138;
    --dm-navy-2: #12314F;
    --dm-teal: #2DD4BF;
    --dm-teal-strong: #14B8A6;
    --dm-blue: #2563EB;
    --dm-orange: #F59E0B;
    --dm-red: #DC2626;

    --dm-page: #F7FAFC;
    --dm-surface: #FFFFFF;
    --dm-surface-2: #F8FAFC;
    --dm-text: #14213D;
    --dm-text-muted: #52606D;
    --dm-border: #D7E0EA;
    --dm-shadow: rgba(15, 23, 42, 0.08);

    --dm-normal-bg: #EFF6FF;
    --dm-normal-text: #123A63;
    --dm-anomaly-bg: #FFF7ED;
    --dm-anomaly-text: #7C2D12;
}

/* Dark/system-dark palette */
@media (prefers-color-scheme: dark) {
    :root {
        --dm-page: #0B1220;
        --dm-surface: #111827;
        --dm-surface-2: #172033;
        --dm-text: #F3F7FB;
        --dm-text-muted: #C5D0DC;
        --dm-border: #344155;
        --dm-shadow: rgba(0, 0, 0, 0.24);

        --dm-normal-bg: #10233B;
        --dm-normal-text: #D7E9FF;
        --dm-anomaly-bg: #3A2411;
        --dm-anomaly-text: #FFE3B3;
    }
}

/* Gradio may also set .dark explicitly, so support that too */
.dark {
    --dm-page: #0B1220;
    --dm-surface: #111827;
    --dm-surface-2: #172033;
    --dm-text: #F3F7FB;
    --dm-text-muted: #C5D0DC;
    --dm-border: #344155;
    --dm-shadow: rgba(0, 0, 0, 0.24);

    --dm-normal-bg: #10233B;
    --dm-normal-text: #D7E9FF;
    --dm-anomaly-bg: #3A2411;
    --dm-anomaly-text: #FFE3B3;
}

/* ---------- global ---------- */
html,
body,
.gradio-container {
    background: var(--dm-page) !important;
    color: var(--dm-text) !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
}

.gradio-container,
.gradio-container p,
.gradio-container span,
.gradio-container div,
.gradio-container label,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6 {
    color: var(--dm-text);
}

.gradio-container a {
    color: var(--dm-blue);
}

#instructions,
#instructions p {
    color: var(--dm-text-muted) !important;
    opacity: 1 !important;
}

/* ---------- DistriMuSe hero ---------- */
.dm-hero {
    position: relative;
    color: #FFFFFF;
    border-radius: 22px;
    margin: 8px 0 18px;
    box-shadow: 0 18px 40px rgba(10, 25, 47, 0.20);
    overflow: hidden;
    isolation: isolate;
}

.dm-hero-image {
    min-height: 285px;
    background-image: var(--hero-image);
    background-size: cover;
    background-position: center right;
    background-repeat: no-repeat;
    background-color: #071A2D;
}

.dm-hero-overlay {
    position: absolute;
    inset: 0;
    z-index: 0;
    background:
        linear-gradient(
            90deg,
            rgba(7, 26, 45, 0.98) 0%,
            rgba(39, 1, 60, 0.95) 42%,
            rgba(88, 0, 99, 0.76) 66%,
            rgba(135, 0, 153, 0.28) 100%
        ),
        linear-gradient(
            180deg,
            rgba(0, 0, 0, 0.06) 0%,
            rgba(0, 0, 0, 0.25) 100%
        );
}

.dm-hero-layout {
    position: relative;
    z-index: 1;
    min-height: 285px;
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(230px, .65fr);
    gap: 28px;
    align-items: center;
    padding: 30px 34px;
}

.dm-hero * {
    color: #FFFFFF !important;
}

.dm-hero-inner {
    max-width: 760px;
}

.dm-eyebrow {
    color: #fdc4ff !important;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .10em;
    font-size: .78rem;
    margin-bottom: 9px;
}

.dm-hero h1 {
    margin: 0 0 12px !important;
    font-size: clamp(2rem, 3.3vw, 3rem) !important;
    line-height: 1.04 !important;
    font-weight: 850 !important;
    letter-spacing: -0.035em;
    text-shadow: 0 2px 14px rgba(0, 0, 0, .22);
}

.dm-hero h1 span {
    display: block;
    color: #F2C8FF !important;
}

.dm-hero p {
    max-width: 790px;
    margin: 0;
    color: rgba(255,255,255,.93) !important;
    font-size: 1rem;
    line-height: 1.65;
    text-shadow: 0 1px 8px rgba(0,0,0,.22);
}

.dm-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 18px;
}

.dm-badges span {
    color: #FFFFFF !important;
    border: 1px solid rgba(235, 166, 255, .72);
    background: rgba(82, 0, 96, .42);
    backdrop-filter: blur(8px);
    border-radius: 999px;
    padding: 6px 11px;
    font-size: .82rem;
    font-weight: 750;
}

.dm-hero-features {
    justify-self: end;
    width: min(100%, 250px);
    display: grid;
    gap: 10px;
}

.dm-hero-feature {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 44px;
    padding: 9px 12px;
    border: 1px solid rgba(255,255,255,.24);
    background: rgba(22, 6, 40, .44);
    backdrop-filter: blur(9px);
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,.08);
    font-size: .9rem;
    font-weight: 700;
}

.feature-icon {
    width: 27px;
    height: 27px;
    display: inline-grid;
    place-items: center;
    flex: 0 0 27px;
    border-radius: 8px;
    background: rgba(246, 196, 255, .15);
    border: 1px solid rgba(246, 196, 255, .32);
    color: #fdc4ff !important;
    font-size: 1rem;
}

/* ---------- generic cards / panels ---------- */
.study-card {
    background: var(--dm-surface) !important;
    color: var(--dm-text) !important;
    border: 1px solid var(--dm-border) !important;
    border-radius: 16px;
    padding: 17px 18px;
    margin: 10px 0 16px 0;
    box-shadow: 0 7px 22px var(--dm-shadow);
}

.study-card-title {
    color: var(--dm-text) !important;
    font-size: 1.05rem;
    font-weight: 800;
}

.study-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 12px;
}

.study-stat {
    background: var(--dm-surface-2) !important;
    color: var(--dm-text) !important;
    border: 1px solid var(--dm-border) !important;
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
}

.study-stat .value {
    color: var(--dm-text) !important;
    font-size: 1.48rem;
    font-weight: 800;
}

.study-stat .label,
.study-meta {
    color: var(--dm-text-muted) !important;
    opacity: 1 !important;
}

.study-stat .label {
    font-size: 0.9rem;
}

.study-meta {
    margin-top: 8px;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* ---------- participant progress ---------- */
.progress-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
}

.progress-count {
    color: var(--dm-text) !important;
    font-size: 1.05rem;
    font-weight: 800;
}

.progress-shell {
    width: 100%;
    height: 13px;
    background: var(--dm-border) !important;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--dm-blue), var(--dm-teal)) !important;
    border-radius: 999px;
}


/* Hidden browser-to-server trigger used when video playback reaches the end. */
#video-finished-trigger {
    display: none !important;
}

.reference-locked {
    background: var(--dm-surface-2) !important;
    border-color: #64748B !important;
    color: var(--dm-text) !important;
}

.reference-locked * {
    color: var(--dm-text) !important;
}

.reference-details {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.reference-details span {
    border: 1px solid currentColor;
    border-radius: 999px;
    padding: 5px 9px;
    font-size: .84rem;
    font-weight: 650;
    opacity: .92;
}


/* ---------- normal/anomaly reference ---------- */
.sequence-reference {
    border-radius: 14px;
    padding: 14px 16px;
    margin: 8px 0 10px;
    border: 1px solid;
    border-left-width: 5px;
}

.reference-kicker {
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .08em;
    opacity: .82;
}

.reference-title {
    font-size: 1.04rem;
    font-weight: 850;
    margin-top: 2px;
}

.reference-text {
    margin-top: 5px;
    line-height: 1.5;
    font-size: .94rem;
}

.reference-normal {
    background: var(--dm-normal-bg) !important;
    border-color: #3B82F6 !important;
    color: var(--dm-normal-text) !important;
}

.reference-normal * {
    color: var(--dm-normal-text) !important;
}

.reference-anomaly {
    background: var(--dm-anomaly-bg) !important;
    border-color: var(--dm-orange) !important;
    color: var(--dm-anomaly-text) !important;
}

.reference-anomaly * {
    color: var(--dm-anomaly-text) !important;
}

.reference-unknown {
    background: var(--dm-surface-2) !important;
    border-color: #94A3B8 !important;
    color: var(--dm-text) !important;
}

/* ---------- video ---------- */
#study-video {
    border: 1px solid var(--dm-border);
    border-radius: 14px !important;
    overflow: hidden;
    background: #020617 !important;
    box-shadow: 0 8px 22px var(--dm-shadow);
    margin-top: 4px !important;
    margin-bottom: 10px !important;
}

#study-video video {
    width: 100% !important;
    height: auto !important;
    max-height: none !important;
    object-fit: contain !important;
    display: block !important;
}

#study-video .wrap,
#study-video [data-testid="video"] {
    min-height: 0 !important;
    height: auto !important;
    aspect-ratio: auto !important;
}

#questions-panel {
    pointer-events: auto !important;
    position: relative;
    z-index: 2;
}

/* ---------- response chips ---------- */
.response-key {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 4px 0 12px 0;
}

.response-chip {
    color: var(--dm-text) !important;
    background: var(--dm-surface-2) !important;
    border: 1px solid var(--dm-border);
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 0.9rem;
    font-weight: 700;
}

.response-chip.chip-normal {
    border-color: #60A5FA;
    background: var(--dm-normal-bg) !important;
    color: var(--dm-normal-text) !important;
}

.response-chip.chip-anomaly {
    border-color: #F59E0B;
    background: var(--dm-anomaly-bg) !important;
    color: var(--dm-anomaly-text) !important;
}

/* ---------- Gradio form controls ---------- */
#questions-panel,
#questions-panel > div,
.gradio-container .form,
.gradio-container .panel {
    color: var(--dm-text) !important;
}

#questions-panel label,
.gradio-container label {
    color: var(--dm-text) !important;
    opacity: 1 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: var(--dm-surface) !important;
    color: var(--dm-text) !important;
    border-color: var(--dm-border) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: var(--dm-text-muted) !important;
    opacity: .8 !important;
}

/* Radio option cards */
#questions-panel [role="radiogroup"] label {
    background: var(--dm-surface-2) !important;
    color: var(--dm-text) !important;
    border-color: var(--dm-border) !important;
    border-radius: 10px !important;
}

#questions-panel [role="radiogroup"] label:hover {
    border-color: var(--dm-teal-strong) !important;
}

/* Radio controls: visual styling only.
   Do not override pointer-events or Gradio's internal interaction rules. */
#q1-map-indication [role="radiogroup"] label,
#q2-right-reason [role="radiogroup"] label,
#q3-right-area [role="radiogroup"] label {
    cursor: pointer;
}

#q1-map-indication input[type="radio"],
#q2-right-reason input[type="radio"],
#q3-right-area input[type="radio"] {
    accent-color: var(--dm-teal-strong);
}


/* Sliders */
.gradio-container input[type="range"] {
    accent-color: var(--dm-teal-strong);
}

/* Buttons */
.gradio-container button {
    font-weight: 750 !important;
}

.gradio-container button.primary {
    background: linear-gradient(135deg, #2563EB, #1557B0) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
}

.gradio-container button.secondary,
.gradio-container button:not(.primary) {
    color: var(--dm-text) !important;
}

/* Accordions and researcher tools */
.gradio-container details,
.gradio-container .accordion {
    background: var(--dm-surface) !important;
    color: var(--dm-text) !important;
    border-color: var(--dm-border) !important;
}

/* ---------- footer ---------- */
.dm-footer {
    margin-top: 26px;
    padding: 18px 20px;
    border-radius: 16px;
    background: #580063;
    color: rgba(255,255,255,.88) !important;
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    font-size: .88rem;
}

.dm-footer * {
    color: rgba(255,255,255,.88) !important;
}

.dm-footer strong {
    color: #FFFFFF !important;
}

.dm-footer-links {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.dm-footer a {
    color: #ffc5f8 !important;
    text-decoration: none;
    font-weight: 750;
}



/* ---------- compact 1–5 radio scales ---------- */
.likert-radio [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(5, minmax(54px, 1fr)) !important;
    gap: 8px !important;
    width: 100% !important;
}

.likert-radio [role="radiogroup"] label {
    min-height: 44px;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    border: 1px solid var(--dm-border) !important;
    border-radius: 10px !important;
    background: var(--dm-surface-2) !important;
    color: var(--dm-text) !important;
    font-weight: 800 !important;
    cursor: pointer;
    padding: 7px 6px !important;
}

.likert-radio [role="radiogroup"] label:hover {
    border-color: var(--dm-teal-strong) !important;
}

.likert-radio input[type="radio"] {
    accent-color: var(--dm-teal-strong);
}

.likert-endpoints {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: var(--dm-text-muted) !important;
    font-size: .82rem;
    margin: -3px 2px 10px 2px;
}

.likert-endpoints span {
    color: var(--dm-text-muted) !important;
}

@media (max-width: 560px) {
    .likert-radio [role="radiogroup"] {
        grid-template-columns: repeat(5, minmax(42px, 1fr)) !important;
        gap: 5px !important;
    }

    .likert-radio [role="radiogroup"] label {
        min-height: 40px;
        padding: 5px 3px !important;
    }

    .likert-endpoints {
        font-size: .76rem;
    }
}

/* ---------- required-field validation ---------- */
.validation-alert {
    background: #FFF7ED !important;
    color: #7C2D12 !important;
    border: 1px solid #FDBA74 !important;
    border-left: 5px solid #EA580C !important;
    border-radius: 12px;
    padding: 12px 14px;
    margin: 8px 0 12px;
}

.validation-alert * {
    color: #7C2D12 !important;
}

.validation-title {
    font-weight: 800;
    margin-bottom: 5px;
}

.validation-alert ul {
    margin: 5px 0 0 18px;
    padding: 0;
}

.field-required {
    color: #B91C1C !important;
    font-size: .86rem;
    font-weight: 800;
    margin: -2px 0 10px 2px;
}

.dark .field-required {
    color: #FCA5A5 !important;
}

.dark .validation-alert {
    background: #3A1C0B !important;
    border-color: #9A3412 !important;
}

.dark .validation-alert,
.dark .validation-alert * {
    color: #FED7AA !important;
}

@media (prefers-color-scheme: dark) {
    .field-required {
        color: #FCA5A5 !important;
    }

    .validation-alert {
        background: #3A1C0B !important;
        border-color: #9A3412 !important;
    }

    .validation-alert,
    .validation-alert * {
        color: #FED7AA !important;
    }
}



/* ---------- categorized event distribution ---------- */
.event-distribution {
    margin-top: 18px;
}

.event-category-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 8px;
}

.event-category-card {
    border: 1px solid var(--dm-border);
    border-radius: 14px;
    padding: 14px;
}

.event-normal-card {
    background: var(--dm-normal-bg);
    border-color: #60A5FA;
}

.event-unexpected-card {
    background: var(--dm-anomaly-bg);
    border-color: #F59E0B;
}

.event-other-card {
    background: var(--dm-surface-2);
}

.event-category-heading {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 9px;
    font-weight: 850;
}

.event-total {
    font-size: .82rem;
    font-weight: 750;
    opacity: .78;
    white-space: nowrap;
}

.event-normal-card .event-category-heading,
.event-normal-card .event-item,
.event-normal-card .event-total {
    color: var(--dm-normal-text) !important;
}

.event-unexpected-card .event-category-heading,
.event-unexpected-card .event-item,
.event-unexpected-card .event-total {
    color: var(--dm-anomaly-text) !important;
}

.event-item {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(127, 127, 127, .22);
}

.event-item:last-child {
    border-bottom: 0;
}

.event-item strong {
    min-width: 24px;
    text-align: right;
}

@media (max-width: 760px) {
    .event-category-grid {
        grid-template-columns: 1fr;
    }
}

/* ---------- Questionnaire / Researcher Tools pages ---------- */
#main-tabs {
    margin-top: 8px;
}

#main-tabs button[role="tab"] {
    font-weight: 800 !important;
    border-radius: 12px 12px 0 0 !important;
    min-height: 46px;
}

#main-tabs button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #580063, #9f01be) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
}

#main-tabs button[role="tab"][aria-selected="false"] {
    background: var(--dm-surface-2) !important;
    color: var(--dm-text) !important;
}

.researcher-hero {
    background: linear-gradient(135deg, #27013c 0%, #9f01be 100%);
    border-radius: 18px;
    padding: 22px 24px;
    margin: 8px 0 16px;
    box-shadow: 0 12px 28px var(--dm-shadow);
}

.researcher-hero h2 {
    color: #FFFFFF !important;
    margin: 3px 0 7px !important;
    font-size: 1.65rem !important;
}

.researcher-hero p {
    color: rgba(255,255,255,.88) !important;
    margin: 0;
    max-width: 900px;
}

.summary-list {
    margin-top: 14px;
    border: 1px solid var(--dm-border);
    border-radius: 12px;
    padding: 12px 14px;
    background: var(--dm-surface-2);
}

.summary-list-title {
    color: var(--dm-text) !important;
    font-weight: 800;
    margin-bottom: 7px;
}

.summary-row {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    padding: 7px 0;
    border-bottom: 1px solid var(--dm-border);
}

.summary-row:last-child {
    border-bottom: 0;
}

.distribution-section {
    margin-top: 16px;
}

.distribution-row {
    display: grid;
    grid-template-columns: minmax(165px, 1fr) minmax(180px, 2fr) auto;
    gap: 10px;
    align-items: center;
    margin: 8px 0;
}

.distribution-bar-shell {
    height: 10px;
    background: var(--dm-border);
    border-radius: 999px;
    overflow: hidden;
}

.distribution-bar {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--dm-blue), var(--dm-teal));
}

.researcher-divider {
    height: 1px;
    background: var(--dm-border);
    margin: 24px 0;
}

@media (max-width: 760px) {
    .distribution-row {
        grid-template-columns: 1fr;
    }
}

/* ---------- responsive ---------- */
@media (max-width: 800px) {
    .dm-hero {
        border-radius: 18px;
    }

    .dm-hero-image {
        min-height: 0;
        background-position: 68% center;
    }

    .dm-hero-layout {
        min-height: 0;
        grid-template-columns: 1fr;
        padding: 24px 22px;
    }

    .dm-hero-overlay {
        background:
            linear-gradient(
                90deg,
                rgba(7, 26, 45, .98) 0%,
                rgba(39, 1, 60, .94) 58%,
                rgba(88, 0, 99, .62) 100%
            );
    }

    .dm-hero-features {
        display: none;
    }

    .dm-hero h1 {
        font-size: clamp(1.65rem, 7vw, 2.3rem) !important;
    }

    .study-grid {
        grid-template-columns: 1fr;
    }

    .gradio-container {
        padding-left: 10px !important;
        padding-right: 10px !important;
    }
}
"""

with gr.Blocks(
    title="Explainable Anomaly Detection User Study",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate",
        radius_size="md",
        text_size="md",
    ),
) as demo:
    with gr.Tabs(elem_id="main-tabs"):
        with gr.Tab("Questionnaire", id="questionnaire"):
            gr.HTML(
                f"""
                <section
                    class="dm-hero dm-hero-image"
                    style="--hero-image: url('{HEADER_IMAGE_URI}');"
                >
                    <div class="dm-hero-overlay"></div>

                    <div class="dm-hero-layout">
                        <div class="dm-hero-inner">
                            <div class="dm-eyebrow">
                                DistriMuSe · UC3 · Safe Interaction with Robots
                            </div>

                            <h1>
                                Explainable Unexpected-Condition
                                <span>Detection Study</span>
                            </h1>

                            <p>
                                Human evaluation of anomaly maps for trustworthy,
                                transparent and safer human–robot collaboration in
                                industrial environments.
                            </p>

                            <div class="dm-badges">
                                <span>DistriMuSe</span>
                                <span>University of Torino</span>
                                <span>Explainable AI</span>
                                <span>Human–Robot Safety</span>
                            </div>
                        </div>

                        <div class="dm-hero-features" aria-label="Study focus">
                            <div class="dm-hero-feature">
                                <span class="feature-icon">◎</span>
                                <span>Human Safety</span>
                            </div>
                            <div class="dm-hero-feature">
                                <span class="feature-icon">⚙</span>
                                <span>Robot Collaboration</span>
                            </div>
                            <div class="dm-hero-feature">
                                <span class="feature-icon">◇</span>
                                <span>Explainable AI</span>
                            </div>
                            <div class="dm-hero-feature">
                                <span class="feature-icon">▥</span>
                                <span>Industrial Impact</span>
                            </div>
                        </div>
                    </div>
                </section>
                """
            )

            gr.Markdown(
                """
                **Purpose.** This study evaluates whether the anomaly/explanation map
                highlights an unexpected condition for the **right reason** and in the
                **right safety area**.

                Please judge the explanation shown in the video, not merely whether an
                anomaly detector produced an alert. Your responses are stored under the
                anonymous participant ID generated for this study.
                """,
                elem_id="instructions",
            )

            try:
                _initial_metadata = load_metadata()
                _overview = study_overview_html(_initial_metadata)
            except Exception as _overview_exc:
                _overview = (
                    "<div class='study-card'>"
                    f"Could not load study overview: {_overview_exc}"
                    "</div>"
                )

            study_overview = gr.HTML(_overview)

            state = gr.State()

            gr.Markdown(
                """
                ### Participant access

                For a **new participant**, click **Start new participant**. The app
                generates an anonymous participant ID automatically. Keep that ID if
                you need to resume the study later.
                """
            )

            with gr.Row():
                new_participant_button = gr.Button(
                    "Start new participant",
                    variant="primary",
                    scale=1,
                )
                participant_id = gr.Textbox(
                    label="Participant ID",
                    placeholder="Generated automatically, e.g. P-A1B2C3",
                    scale=2,
                )
                resume_button = gr.Button("Resume participant", scale=1)

            status = gr.Markdown(
                "Start a new participant or enter an existing ID to resume."
            )
            participant_progress = gr.HTML("")
            sequence_reference = gr.HTML("")

            video = gr.Video(
                label="Study clip",
                interactive=False,
                visible=False,
                elem_id="study-video",
            )

            # Programmatically clicked by the browser only after the video reaches its end.
            video_finished_trigger = gr.Button(
                "Video finished",
                elem_id="video-finished-trigger",
                visible=True,
            )

            validation_alert = gr.HTML("")

            with gr.Group(visible=False, elem_id="questions-panel") as questions_group:
                gr.HTML(
                    """
                    <div class="response-key">
                        <span class="response-chip chip-normal">🔵 Normal</span>
                        <span class="response-chip chip-anomaly">🟠 Unexpected</span>
                        <span class="response-chip">🟢 Yes</span>
                        <span class="response-chip">🟡 Partially</span>
                        <span class="response-chip">🔴 No</span>
                        <span class="response-chip">⚪ Cannot determine</span>
                    </div>
                    <div class="study-meta" style="margin-bottom:8px;">
                        Select one response for each question below.
                    </div>
                    """
                )

                map_indicates_unexpected = gr.Radio(
                    MAP_INDICATION_OPTIONS,
                    label=(
                        "1. What does the explanation map indicate for this sequence?"
                    ),
                    interactive=True,
                    elem_id="q1-map-indication",
                )
                q1_error = gr.HTML("")

                right_reason = gr.Radio(
                    QUESTION_OPTIONS,
                    label=(
                        "2. Does the highlighted region correspond to the actual cause "
                        "of the unexpected condition?"
                    ),
                    interactive=True,
                    elem_id="q2-right-reason",
                )
                q2_error = gr.HTML("")

                right_area = gr.Radio(
                    QUESTION_OPTIONS,
                    label=(
                        "3. Is the unexpected condition highlighted in the correct "
                        "safety area?"
                    ),
                    interactive=True,
                    elem_id="q3-right-area",
                )
                q3_error = gr.HTML("")

                localization_score = gr.Radio(
                    choices=[
                        ("1", 1),
                        ("2", 2),
                        ("3", 3),
                        ("4", 4),
                        ("5", 5),
                    ],
                    value=None,
                    interactive=True,
                    elem_id="q4-localization",
                    elem_classes=["likert-radio"],
                    label=(
                        "4. How well does the anomaly map localize the actual cause? "
                        "(1 = very poor, 5 = excellent)"
                    ),
                )
                gr.HTML(
                    """
                    <div class="likert-endpoints">
                        <span>1 — Very poor / misleading</span>
                        <span>5 — Excellent / very appropriate</span>
                    </div>
                    """
                )
                q4_error = gr.HTML("")

                confidence = gr.Radio(
                    choices=[
                        ("1", 1),
                        ("2", 2),
                        ("3", 3),
                        ("4", 4),
                        ("5", 5),
                    ],
                    value=None,
                    interactive=True,
                    elem_id="q5-confidence",
                    elem_classes=["likert-radio"],
                    label=(
                        "5. How confident are you in your assessment? "
                        "(1 = very low, 5 = very high)"
                    ),
                )
                gr.HTML(
                    """
                    <div class="likert-endpoints">
                        <span>1 — Very low</span>
                        <span>5 — Very high</span>
                    </div>
                    """
                )
                q5_error = gr.HTML("")

                comment = gr.Textbox(
                    label="Optional comment",
                    lines=3,
                    interactive=True,
                    placeholder="Optional: explain what looks correct, incorrect, or ambiguous.",
                )

            submit_button = gr.Button(
                "Submit & Next",
                variant="primary",
                visible=False,
            )

            gr.HTML(
                """
                <footer class="dm-footer">
                    <div>
                        <strong>DistriMuSe UC3 · Explainable AI User Study</strong><br>
                        University of Torino · Safe interaction and cooperation with robots
                    </div>
                    <div class="dm-footer-links">
                        <a href="https://distrimuse.eu/" target="_blank">DistriMuSe project</a>
                        <a href="https://rashidrao-pk.github.io/projects/advis-distrimuse-sr/" target="_blank">ADVIS project page</a>
                        <a href="https://github.com/rashidrao-pk/advis_distrimuse_unito_SR" target="_blank">GitHub</a>
                    </div>
                </footer>
                """
            )

        with gr.Tab("Researcher Tools", id="researcher-tools"):
            gr.HTML(
                """
                <section class="researcher-hero">
                    <div class="dm-eyebrow">DistriMuSe · UC3 · Research Dashboard</div>
                    <h2>Researcher Tools & Study Summary</h2>
                    <p>
                        Review questionnaire completion and explanation-quality
                        indicators, inspect response distributions, export the
                        collected data, and run optional GPU video QA.
                    </p>
                </section>
                """
            )

            researcher_summary = gr.HTML(researcher_summary_html())
            response_summary = gr.HTML(researcher_response_summary())
            response_breakdown = gr.HTML(response_breakdown_html())

            gr.Markdown("### Participant progress")
            participant_table = gr.Dataframe(
                value=participant_progress_table(),
                interactive=False,
                wrap=True,
                label="Current participant completion",
            )

            with gr.Row():
                refresh_summary_button = gr.Button("Refresh dashboard", variant="primary")
                export_button = gr.Button("Export responses as CSV")

            export_file = gr.File(label="CSV export", interactive=False)

            gr.Markdown(
                """
                ### How effective are the explanations?

                The dashboard reports several complementary indicators instead of
                collapsing them into one arbitrary score:

                - **Map/reference agreement**: whether participants interpret the map
                  as Normal vs. Unexpected consistently with the curated sequence.
                - **Right reason**: whether the highlighted explanation corresponds
                  to the actual cause of an anomaly.
                - **Right area**: whether the highlighted explanation is spatially
                  located in the relevant safety area.
                - **Localization rating**: perceived spatial precision of the map.
                - **Normal-sequence avoidance**: whether false anomaly highlights are
                  avoided when no unexpected event is expected.
                - **Confidence**: how certain participants are about their judgments.
                """
            )

            gr.HTML('<div class="researcher-divider"></div>')

            gr.Markdown(
                """
                ### GPU Video Quality Check

                This researcher-only utility is placed at the end of this page.
                It samples the middle frame of a selected study video and performs
                a small CUDA-based brightness/contrast check. It does not affect
                questionnaire responses.
                """
            )

            try:
                _qa_video_choices = [row["filename"] for row in load_metadata()]
            except Exception:
                _qa_video_choices = []

            qa_video_dropdown = gr.Dropdown(
                choices=_qa_video_choices,
                value=_qa_video_choices[0] if _qa_video_choices else None,
                label="Study video",
            )
            qa_video_button = gr.Button("Run GPU quality check")
            qa_video_result = gr.Markdown("")

    demo.load(
        fn=None,
        js='\n() => {\n    const TRIGGER_ID = "video-finished-trigger";\n\n    function bindVideo(video) {\n        if (!video || video.dataset.studyEndBound === "1") return;\n\n        video.dataset.studyEndBound = "1";\n\n        video.addEventListener("loadedmetadata", () => {\n            // A new trial/source has loaded.\n            video.dataset.studyUnlocked = "0";\n        });\n\n        video.addEventListener("ended", () => {\n            if (video.dataset.studyUnlocked === "1") return;\n            video.dataset.studyUnlocked = "1";\n\n            const trigger = document.querySelector(`#${TRIGGER_ID} button`)\n                || document.getElementById(TRIGGER_ID);\n\n            if (trigger) {\n                trigger.click();\n            }\n        });\n    }\n\n    function scan() {\n        const container = document.getElementById("study-video");\n        if (!container) return;\n        const video = container.querySelector("video");\n        if (video) bindVideo(video);\n    }\n\n    scan();\n\n    const observer = new MutationObserver(() => scan());\n    observer.observe(document.body, {\n        childList: true,\n        subtree: true,\n        attributes: true,\n        attributeFilter: ["src"]\n    });\n}\n',
    )

    new_participant_button.click(
        start_new_participant,
        outputs=[
            participant_id,
            state,
            video,
            status,
            participant_progress,
            sequence_reference,
            questions_group,
            submit_button,
        ],
    )

    resume_button.click(
        resume_study,
        inputs=[participant_id],
        outputs=[
            participant_id,
            state,
            video,
            status,
            participant_progress,
            sequence_reference,
            questions_group,
            submit_button,
        ],
    )

    video_finished_trigger.click(
        unlock_questions_after_video,
        inputs=[state],
        outputs=[
            state,
            sequence_reference,
            questions_group,
            submit_button,
            right_reason,
            right_area,
            localization_score,
        ],
    )

    submit_button.click(
        submit_response,
        inputs=[
            state,
            map_indicates_unexpected,
            right_reason,
            right_area,
            localization_score,
            confidence,
            comment,
        ],
        outputs=[
            state,
            video,
            status,
            participant_progress,
            sequence_reference,
            validation_alert,
            q1_error,
            q2_error,
            q3_error,
            q4_error,
            q5_error,
            map_indicates_unexpected,
            right_reason,
            right_area,
            localization_score,
            confidence,
            comment,
            questions_group,
            submit_button,
        ],
    )

    qa_video_button.click(
        gpu_video_quality_check,
        inputs=[qa_video_dropdown],
        outputs=qa_video_result,
    )

    refresh_summary_button.click(
        refresh_researcher_dashboard,
        outputs=[
            researcher_summary,
            response_summary,
            response_breakdown,
            participant_table,
        ],
    )

    export_button.click(
        export_local_csv,
        outputs=export_file,
    )

if __name__ == "__main__":
    demo.launch()
