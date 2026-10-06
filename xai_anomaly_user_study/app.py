
import base64
import csv
import cv2
import html as html_lib
import io
import os
import re
import secrets
import shutil
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

DB_MANAGER_PASSWORD = os.getenv("DB_MANAGER_PASSWORD", "").strip()

DB_LOCK = threading.Lock()
HF_LOCK = threading.Lock()

QUESTION_OPTIONS = [
    "Yes",
    "No",
]

MAP_INDICATION_OPTIONS = QUESTION_OPTIONS.copy()

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
    "Yes": "Yes",
    "No": "No",
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_participants (
                participant_id TEXT PRIMARY KEY,
                created_at_utc TEXT,
                deleted_at_utc TEXT NOT NULL,
                deletion_reason TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deleted_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT NOT NULL,
                trial_index INTEGER,
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
                submitted_at_utc TEXT,
                deleted_at_utc TEXT NOT NULL,
                deletion_reason TEXT
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

def ordered_trials(participant_id: str, metadata):
    """
    Keep study videos in ascending video_id order for every participant.

    Natural sorting is used so IDs such as V2 come before V10.
    The participant_id argument is retained for call-site compatibility.
    """
    def natural_video_id_key(row):
        video_id = str(row.get("video_id", ""))
        parts = re.split(r"(\d+)", video_id)
        return [
            int(part) if part.isdigit() else part.lower()
            for part in parts
        ]

    return sorted(list(metadata), key=natural_video_id_key)



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
    </div>
    """


def participant_progress_html(participant_id: str, metadata):
    total = len(metadata)
    completed = len(completed_video_ids(participant_id))
    remaining = max(0, total - completed)
    percent = 0 if total == 0 else round((completed / total) * 100)

    return f"""
    <div class="study-card compact-progress-card">
        <div class="compact-progress-row">
            <div class="compact-progress-title">Progress</div>
            <div class="compact-progress-item">
                <strong>{completed} / {total}</strong>
                <span>Completed</span>
            </div>
            <div class="compact-progress-item">
                <strong>{remaining}</strong>
                <span>Remaining</span>
            </div>
        </div>

        <div class="progress-shell compact-progress-shell">
            <div class="progress-fill" style="width:{percent}%"></div>
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
    ordered = ordered_trials(participant_id, metadata)
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
    <div class="sequence-reference reference-locked compact-lock">
        <div class="reference-title">🔒 Watch the full video to continue</div>
    </div>
    """


def sequence_reference_html(current):
    """
    Show a compact, collapsible study reference after the participant watches
    the full clip. The panel is collapsed by default to keep the questionnaire
    visually focused.
    """
    gt = (current.get("ground_truth", "") or "").strip().lower()
    event_type = html_lib.escape(format_event_type(current.get("event_type", "")))
    area = html_lib.escape(format_area_name(current.get("expected_area", "")))
    filename = html_lib.escape(str(current.get("filename", "") or ""))

    if gt in {"normal", "expected", "no_anomaly", "no anomaly"}:
        return f"""
        <details class="sequence-reference reference-normal reference-collapsible">
            <summary>
                <span class="reference-summary-label">Study reference</span>
                <span class="reference-summary-state">✓ Normal workflow</span>
                <span class="reference-summary-toggle">Show details</span>
            </summary>
            <div class="reference-body">
                <div class="reference-details compact-reference-details">
                    <span><strong>Video:</strong> {filename}</span>
                    <span><strong>Condition:</strong> {event_type}</span>
                    <span><strong>Expected anomalous area:</strong> None</span>
                </div>
                <div class="reference-text compact-reference-text">
                    No unexpected condition is expected. The explanation should not
                    falsely highlight an anomalous cause or safety area.
                </div>
            </div>
        </details>
        """

    if gt in {"unexpected", "anomalous", "anomaly", "abnormal"}:
        return f"""
        <details class="sequence-reference reference-anomaly reference-collapsible">
            <summary>
                <span class="reference-summary-label">Study reference</span>
                <span class="reference-summary-state">! Unexpected / anomalous</span>
                <span class="reference-summary-toggle">Show details</span>
            </summary>
            <div class="reference-body">
                <div class="reference-details compact-reference-details">
                    <span><strong>Video:</strong> {filename}</span>
                    <span><strong>Anomalous condition:</strong> {event_type}</span>
                    <span><strong>Relevant safety area:</strong> {area}</span>
                </div>
                <div class="reference-text compact-reference-text">
                    An unexpected condition is present. Judge whether the highlighted
                    explanation corresponds to the actual cause.
                </div>
            </div>
        </details>
        """

    return f"""
    <details class="sequence-reference reference-unknown reference-collapsible">
        <summary>
            <span class="reference-summary-label">Study reference</span>
            <span class="reference-summary-state">? Reference not specified</span>
            <span class="reference-summary-toggle">Show details</span>
        </summary>
        <div class="reference-body">
            <div class="reference-details compact-reference-details">
                <span><strong>Video:</strong> {filename}</span>
                <span><strong>Condition:</strong> {event_type}</span>
                <span><strong>Area:</strong> {area}</span>
            </div>
        </div>
    </details>
    """



def question_labels_for_current(current):
    return (
        "1. Is the system behavior shown in this video consistent with the description above?",
        "2. Does the highlighted area correctly explain the system decision?",
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
        )

    state["video_watched"] = True
    current = state["remaining"][0]
    q1_label, q2_label = question_labels_for_current(current)

    return (
        state,
        sequence_reference_html(current),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(label=q1_label),
        gr.update(label=q2_label),
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
        status = ""
        video_update = gr.update(value=video_path, visible=True)
        reference_html = ""

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
                "Participant ID not found. Use **Start study** to create a new ID.",
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
            None,
            None,
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
            map_indicates_unexpected,
            right_reason,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    fields = [
        ("Question 1 — condition indication", map_indicates_unexpected),
        ("Question 2 — right reason", right_reason),
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
            map_indicates_unexpected,
            right_reason,
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
        "right_area": None,
        "localization_score": None,
        "confidence": None,
        "comment": "",
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
        None,
        None,
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
    completed_participants = int(
        (completed_per_participant >= total_videos).sum()
    ) if total_videos else 0

    possible = participants * total_videos
    coverage = (total_responses / possible * 100) if possible else 0.0

    q1_yes = _percent_of(df["map_indicates_unexpected"], "Yes")
    q1_positive = _percent_of(df["map_indicates_unexpected"], ["Yes", "Partially"])
    q2_yes = _percent_of(df["right_reason"], "Yes")
    q2_positive = _percent_of(df["right_reason"], ["Yes", "Partially"])

    q_counts = {
        "Q1 — System behavior consistent": int(
            df["map_indicates_unexpected"].notna().sum()
        ),
        "Q2 — Highlight explains decision": int(df["right_reason"].notna().sum()),
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
            <div class="study-stat"><div class="value">{q1_yes:.1f}%</div><div class="label">Q1 = Yes</div></div>
            <div class="study-stat"><div class="value">{q2_yes:.1f}%</div><div class="label">Q2 = Yes</div></div>
        </div>

        <div class="study-grid">
            <div class="study-stat"><div class="value">{q1_positive:.1f}%</div><div class="label">Q1 = Yes / Partially</div></div>
            <div class="study-stat"><div class="value">{q2_positive:.1f}%</div><div class="label">Q2 = Yes / Partially</div></div>
        </div>

        <div class="summary-list">
            <div class="summary-list-title">Filled-answer counts</div>
            {q_rows}
        </div>

        <div class="study-meta" style="margin-top:12px;">
            <strong>Effectiveness:</strong> the reduced questionnaire focuses on
            whether the explanation correctly represents the sequence condition
            and whether the highlighted region corresponds to the actual cause.
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

    def distribution(column):
        counts = df[column].fillna("Missing").astype(str).value_counts().to_dict()
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

    return f"""
    <div class="study-card">
        <div class="study-card-title">Response distributions</div>

        <div class="distribution-section">
            <div class="summary-list-title">Q1 — System behavior consistent</div>
            {distribution("map_indicates_unexpected")}
        </div>

        <div class="distribution-section">
            <div class="summary-list-title">Q2 — Highlight explains decision</div>
            {distribution("right_reason")}
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

def _db_connect():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def unlock_database_manager(password):
    """
    Unlock the Database Manager only when the supplied password matches the
    DB_MANAGER_PASSWORD environment variable / Hugging Face Space Secret.
    """
    entered = (password or "").strip()

    if not DB_MANAGER_PASSWORD:
        return (
            """
            <div class="validation-alert">
                <div class="validation-title">Database Manager is not configured</div>
                <div>
                    Add a Hugging Face Space Secret named
                    <code>DB_MANAGER_PASSWORD</code>, then restart the Space.
                </div>
            </div>
            """,
            gr.update(visible=False),
            "",
        )

    if entered != DB_MANAGER_PASSWORD:
        return (
            """
            <div class="validation-alert">
                <div class="validation-title">Incorrect password</div>
                <div>Access to Database Manager was not granted.</div>
            </div>
            """,
            gr.update(visible=False),
            "",
        )

    return (
        """
        <div class="db-unlocked-message">
            ✓ Database Manager unlocked for this browser session.
        </div>
        """,
        gr.update(visible=True),
        "",
    )


def lock_database_manager():
    return (
        """
        <div class="db-locked-message">
            🔒 Database Manager locked.
        </div>
        """,
        gr.update(visible=False),
        "",
    )



def database_status_html():
    """Compact status card for the Database Manager page."""
    with _db_connect() as conn:
        participants = conn.execute(
            "SELECT COUNT(*) AS n FROM participants"
        ).fetchone()["n"]
        responses = conn.execute(
            "SELECT COUNT(*) AS n FROM responses"
        ).fetchone()["n"]
        deleted_participants = conn.execute(
            "SELECT COUNT(*) AS n FROM deleted_participants"
        ).fetchone()["n"]
        deleted_responses = conn.execute(
            "SELECT COUNT(*) AS n FROM deleted_responses"
        ).fetchone()["n"]
        last_row = conn.execute(
            "SELECT MAX(submitted_at_utc) AS last_response FROM responses"
        ).fetchone()
        last_response = last_row["last_response"] or "No responses yet"

    remote_state = (
        f"Configured: {HF_DATA_REPO}"
        if HF_TOKEN and HF_DATA_REPO
        else "Not configured"
    )

    return f"""
    <div class="study-card db-status-card">
        <div class="study-card-title">Database status</div>
        <div class="study-grid">
            <div class="study-stat">
                <div class="value">{participants}</div>
                <div class="label">Active participants</div>
            </div>
            <div class="study-stat">
                <div class="value">{responses}</div>
                <div class="label">Active responses</div>
            </div>
            <div class="study-stat">
                <div class="value">{deleted_participants}</div>
                <div class="label">Archived participants</div>
            </div>
        </div>
        <div class="study-grid">
            <div class="study-stat">
                <div class="value">{deleted_responses}</div>
                <div class="label">Archived responses</div>
            </div>
            <div class="study-stat">
                <div class="value" style="font-size:1rem;">{html_lib.escape(str(last_response))}</div>
                <div class="label">Last response (UTC)</div>
            </div>
            <div class="study-stat">
                <div class="value" style="font-size:1rem;">{html_lib.escape(remote_state)}</div>
                <div class="label">Hugging Face sync</div>
            </div>
        </div>
    </div>
    """


def participant_records_dataframe():
    with _db_connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                p.participant_id AS participant_id,
                p.created_at_utc AS created_at_utc,
                COUNT(DISTINCT r.video_id) AS responses,
                MAX(r.submitted_at_utc) AS last_response_utc
            FROM participants p
            LEFT JOIN responses r
                ON p.participant_id = r.participant_id
            GROUP BY p.participant_id, p.created_at_utc
            ORDER BY p.created_at_utc DESC
            """,
            conn,
        )

    try:
        total_videos = len(load_metadata())
    except Exception:
        total_videos = 0

    if df.empty:
        return pd.DataFrame(
            columns=[
                "Participant",
                "Created (UTC)",
                "Responses",
                "Progress (%)",
                "Last response (UTC)",
            ]
        )

    df["Progress (%)"] = (
        (df["responses"] / total_videos * 100).round(1)
        if total_videos else 0.0
    )
    df = df.rename(
        columns={
            "participant_id": "Participant",
            "created_at_utc": "Created (UTC)",
            "responses": "Responses",
            "last_response_utc": "Last response (UTC)",
        }
    )
    return df[
        ["Participant", "Created (UTC)", "Responses", "Progress (%)", "Last response (UTC)"]
    ]


def active_responses_dataframe():
    with _db_connect() as conn:
        return pd.read_sql_query(
            """
            SELECT
                participant_id AS Participant,
                video_id AS Video,
                event_type AS Event,
                ground_truth AS GroundTruth,
                map_indicates_unexpected AS Q1,
                right_reason AS Q2,
                response_seconds AS ResponseSeconds,
                submitted_at_utc AS SubmittedUTC
            FROM responses
            ORDER BY submitted_at_utc DESC
            """,
            conn,
        )


def archived_participants_dataframe():
    with _db_connect() as conn:
        return pd.read_sql_query(
            """
            SELECT
                participant_id AS Participant,
                created_at_utc AS CreatedUTC,
                deleted_at_utc AS DeletedUTC,
                deletion_reason AS Reason
            FROM deleted_participants
            ORDER BY deleted_at_utc DESC
            """,
            conn,
        )


def participant_choices():
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT participant_id FROM participants ORDER BY participant_id"
        ).fetchall()
    return [row["participant_id"] for row in rows]


def archived_participant_choices():
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT participant_id FROM deleted_participants ORDER BY participant_id"
        ).fetchall()
    return [row["participant_id"] for row in rows]


def response_choices(participant_id):
    if not participant_id:
        return []
    with _db_connect() as conn:
        rows = conn.execute(
            """
            SELECT video_id, event_type
            FROM responses
            WHERE participant_id = ?
            ORDER BY trial_index
            """,
            (participant_id,),
        ).fetchall()
    return [
        (f"{row['video_id']} — {row['event_type'] or 'No event label'}", row["video_id"])
        for row in rows
    ]


def update_response_dropdown(participant_id):
    choices = response_choices(participant_id)
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def backup_database():
    """Create a timestamped SQLite backup using SQLite's backup API."""
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"study_backup_{stamp}.db"

    with sqlite3.connect(SQLITE_PATH) as source:
        with sqlite3.connect(backup_path) as destination:
            source.backup(destination)

    return str(backup_path)


def _remote_responses_dataframe():
    if not (HF_TOKEN and HF_DATA_REPO):
        return None

    from huggingface_hub import hf_hub_download

    try:
        remote_path = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=HF_RESPONSES_FILE,
            token=HF_TOKEN,
        )
        return pd.read_csv(remote_path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)


def _upload_remote_responses(df, commit_message):
    if not (HF_TOKEN and HF_DATA_REPO):
        return False, "Remote persistence is not configured."

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    df = df.reindex(columns=SCHEMA_COLUMNS)
    content = df.to_csv(index=False).encode("utf-8")
    api.upload_file(
        path_or_fileobj=io.BytesIO(content),
        path_in_repo=HF_RESPONSES_FILE,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return True, ""


def delete_remote_participant(participant_id):
    if not (HF_TOKEN and HF_DATA_REPO):
        return True, "Remote persistence not configured; local archive completed."

    try:
        remote_df = _remote_responses_dataframe()
        if remote_df is None or remote_df.empty:
            return True, "No remote response rows found."
        filtered = remote_df[
            remote_df["participant_id"].astype(str) != str(participant_id)
        ].copy()
        _upload_remote_responses(
            filtered,
            f"Delete participant {participant_id} from XAI user study",
        )
        return True, ""
    except Exception as exc:
        return False, str(exc)


def delete_remote_response(participant_id, video_id):
    if not (HF_TOKEN and HF_DATA_REPO):
        return True, "Remote persistence not configured; local archive completed."

    try:
        remote_df = _remote_responses_dataframe()
        if remote_df is None or remote_df.empty:
            return True, "No remote response rows found."
        mask = (
            (remote_df["participant_id"].astype(str) == str(participant_id))
            & (remote_df["video_id"].astype(str) == str(video_id))
        )
        filtered = remote_df[~mask].copy()
        _upload_remote_responses(
            filtered,
            f"Delete response {participant_id}/{video_id}",
        )
        return True, ""
    except Exception as exc:
        return False, str(exc)


def archive_participant(participant_id, confirmation, reason, delete_remote=True):
    participant_id = (participant_id or "").strip()
    confirmation = (confirmation or "").strip()

    if not participant_id:
        return "Select a participant first."

    if confirmation != "DELETE":
        return "Type **DELETE** exactly to confirm participant deletion."

    # Always make a backup before a destructive action.
    backup_path = backup_database()
    now = datetime.now(timezone.utc).isoformat()
    reason = (reason or "").strip() or "Deleted from Database Manager"

    with DB_LOCK, _db_connect() as conn:
        participant = conn.execute(
            """
            SELECT participant_id, created_at_utc
            FROM participants
            WHERE participant_id = ?
            """,
            (participant_id,),
        ).fetchone()

        if not participant:
            return f"Participant `{participant_id}` was not found."

        conn.execute(
            """
            INSERT OR REPLACE INTO deleted_participants
            (participant_id, created_at_utc, deleted_at_utc, deletion_reason)
            VALUES (?, ?, ?, ?)
            """,
            (
                participant["participant_id"],
                participant["created_at_utc"],
                now,
                reason,
            ),
        )

        response_rows = conn.execute(
            "SELECT * FROM responses WHERE participant_id = ?",
            (participant_id,),
        ).fetchall()

        for row in response_rows:
            conn.execute(
                """
                INSERT INTO deleted_responses (
                    participant_id, trial_index, video_id, source_filename,
                    ground_truth, expected_area, event_type,
                    map_indicates_unexpected, right_reason, right_area,
                    localization_score, confidence, comment,
                    response_seconds, submitted_at_utc,
                    deleted_at_utc, deletion_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["participant_id"],
                    row["trial_index"],
                    row["video_id"],
                    row["source_filename"],
                    row["ground_truth"],
                    row["expected_area"],
                    row["event_type"],
                    row["map_indicates_unexpected"],
                    row["right_reason"],
                    row["right_area"],
                    row["localization_score"],
                    row["confidence"],
                    row["comment"],
                    row["response_seconds"],
                    row["submitted_at_utc"],
                    now,
                    reason,
                ),
            )

        conn.execute(
            "DELETE FROM responses WHERE participant_id = ?",
            (participant_id,),
        )
        conn.execute(
            "DELETE FROM participants WHERE participant_id = ?",
            (participant_id,),
        )
        conn.commit()

    remote_note = ""
    if delete_remote:
        ok, msg = delete_remote_participant(participant_id)
        if ok:
            remote_note = " Remote response rows were also removed."
        else:
            remote_note = f" ⚠ Remote deletion failed: {msg}"

    return (
        f"✅ Participant `{participant_id}` and {len(response_rows)} response(s) "
        f"were archived and removed from the active study. "
        f"Backup: `{backup_path}`.{remote_note}"
    )


def archive_single_response(participant_id, video_id, confirmation, reason, delete_remote=True):
    participant_id = (participant_id or "").strip()
    video_id = (video_id or "").strip()
    confirmation = (confirmation or "").strip()

    if not participant_id or not video_id:
        return "Select both a participant and a response."

    if confirmation != "DELETE":
        return "Type **DELETE** exactly to confirm response deletion."

    backup_path = backup_database()
    now = datetime.now(timezone.utc).isoformat()
    reason = (reason or "").strip() or "Response deleted from Database Manager"

    with DB_LOCK, _db_connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM responses
            WHERE participant_id = ? AND video_id = ?
            """,
            (participant_id, video_id),
        ).fetchone()

        if not row:
            return f"Response `{participant_id}/{video_id}` was not found."

        conn.execute(
            """
            INSERT INTO deleted_responses (
                participant_id, trial_index, video_id, source_filename,
                ground_truth, expected_area, event_type,
                map_indicates_unexpected, right_reason, right_area,
                localization_score, confidence, comment,
                response_seconds, submitted_at_utc,
                deleted_at_utc, deletion_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["participant_id"],
                row["trial_index"],
                row["video_id"],
                row["source_filename"],
                row["ground_truth"],
                row["expected_area"],
                row["event_type"],
                row["map_indicates_unexpected"],
                row["right_reason"],
                row["right_area"],
                row["localization_score"],
                row["confidence"],
                row["comment"],
                row["response_seconds"],
                row["submitted_at_utc"],
                now,
                reason,
            ),
        )
        conn.execute(
            """
            DELETE FROM responses
            WHERE participant_id = ? AND video_id = ?
            """,
            (participant_id, video_id),
        )
        conn.commit()

    remote_note = ""
    if delete_remote:
        ok, msg = delete_remote_response(participant_id, video_id)
        if ok:
            remote_note = " Remote response row was also removed."
        else:
            remote_note = f" ⚠ Remote deletion failed: {msg}"

    return (
        f"✅ Response `{participant_id}/{video_id}` was archived and removed. "
        f"Backup: `{backup_path}`.{remote_note}"
    )


def restore_archived_participant(participant_id, confirmation):
    participant_id = (participant_id or "").strip()
    confirmation = (confirmation or "").strip()

    if not participant_id:
        return "Select an archived participant first."

    if confirmation != "RESTORE":
        return "Type **RESTORE** exactly to confirm."

    backup_path = backup_database()

    with DB_LOCK, _db_connect() as conn:
        participant = conn.execute(
            """
            SELECT *
            FROM deleted_participants
            WHERE participant_id = ?
            """,
            (participant_id,),
        ).fetchone()

        if not participant:
            return f"Archived participant `{participant_id}` was not found."

        conn.execute(
            """
            INSERT OR IGNORE INTO participants
            (participant_id, created_at_utc)
            VALUES (?, ?)
            """,
            (participant["participant_id"], participant["created_at_utc"]),
        )

        response_rows = conn.execute(
            """
            SELECT *
            FROM deleted_responses
            WHERE participant_id = ?
            ORDER BY trial_index
            """,
            (participant_id,),
        ).fetchall()

        restored = 0
        for row in response_rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO responses (
                    participant_id, trial_index, video_id, source_filename,
                    ground_truth, expected_area, event_type,
                    map_indicates_unexpected, right_reason, right_area,
                    localization_score, confidence, comment,
                    response_seconds, submitted_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["participant_id"],
                    row["trial_index"],
                    row["video_id"],
                    row["source_filename"],
                    row["ground_truth"],
                    row["expected_area"],
                    row["event_type"],
                    row["map_indicates_unexpected"],
                    row["right_reason"],
                    row["right_area"],
                    row["localization_score"],
                    row["confidence"],
                    row["comment"],
                    row["response_seconds"],
                    row["submitted_at_utc"],
                ),
            )
            restored += 1

        conn.execute(
            "DELETE FROM deleted_responses WHERE participant_id = ?",
            (participant_id,),
        )
        conn.execute(
            "DELETE FROM deleted_participants WHERE participant_id = ?",
            (participant_id,),
        )
        conn.commit()

    remote_note = ""
    if HF_TOKEN and HF_DATA_REPO:
        try:
            sync_responses_to_hf()
            remote_note = " Restored responses were synchronized remotely."
        except Exception as exc:
            remote_note = f" ⚠ Remote re-sync failed: {exc}"

    return (
        f"✅ Restored participant `{participant_id}` with {restored} response(s). "
        f"Backup: `{backup_path}`.{remote_note}"
    )


def reset_entire_study(confirmation, reason, delete_remote=True):
    confirmation = (confirmation or "").strip()

    if confirmation != "DELETE ALL":
        return "Type **DELETE ALL** exactly to reset the active study."

    backup_path = backup_database()
    now = datetime.now(timezone.utc).isoformat()
    reason = (reason or "").strip() or "Full study reset from Database Manager"

    with DB_LOCK, _db_connect() as conn:
        participants = conn.execute(
            "SELECT participant_id, created_at_utc FROM participants"
        ).fetchall()
        responses = conn.execute(
            "SELECT * FROM responses"
        ).fetchall()

        for p in participants:
            conn.execute(
                """
                INSERT OR REPLACE INTO deleted_participants
                (participant_id, created_at_utc, deleted_at_utc, deletion_reason)
                VALUES (?, ?, ?, ?)
                """,
                (p["participant_id"], p["created_at_utc"], now, reason),
            )

        for row in responses:
            conn.execute(
                """
                INSERT INTO deleted_responses (
                    participant_id, trial_index, video_id, source_filename,
                    ground_truth, expected_area, event_type,
                    map_indicates_unexpected, right_reason, right_area,
                    localization_score, confidence, comment,
                    response_seconds, submitted_at_utc,
                    deleted_at_utc, deletion_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["participant_id"],
                    row["trial_index"],
                    row["video_id"],
                    row["source_filename"],
                    row["ground_truth"],
                    row["expected_area"],
                    row["event_type"],
                    row["map_indicates_unexpected"],
                    row["right_reason"],
                    row["right_area"],
                    row["localization_score"],
                    row["confidence"],
                    row["comment"],
                    row["response_seconds"],
                    row["submitted_at_utc"],
                    now,
                    reason,
                ),
            )

        conn.execute("DELETE FROM responses")
        conn.execute("DELETE FROM participants")
        conn.commit()

    remote_note = ""
    if delete_remote and HF_TOKEN and HF_DATA_REPO:
        try:
            empty_df = pd.DataFrame(columns=SCHEMA_COLUMNS)
            _upload_remote_responses(
                empty_df,
                "Reset XAI user study responses",
            )
            remote_note = " Remote responses.csv was also reset."
        except Exception as exc:
            remote_note = f" ⚠ Remote reset failed: {exc}"

    return (
        f"✅ Active study reset: {len(participants)} participant(s) and "
        f"{len(responses)} response(s) archived. Backup: `{backup_path}`."
        f"{remote_note}"
    )


def refresh_database_manager():
    participants = participant_choices()
    archived = archived_participant_choices()

    participant_value = participants[0] if participants else None
    response_opts = response_choices(participant_value)
    response_value = response_opts[0][1] if response_opts else None

    archived_value = archived[0] if archived else None

    return (
        database_status_html(),
        participant_records_dataframe(),
        active_responses_dataframe(),
        archived_participants_dataframe(),
        gr.update(choices=participants, value=participant_value),
        gr.update(choices=response_opts, value=response_value),
        gr.update(choices=archived, value=archived_value),
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


/* ---------- top tab navigation ---------- */
#main-tabs [role="tablist"] {
    position: static !important;
    margin: 0 0 14px !important;
    padding: 0 !important;
    border-bottom: 1px solid var(--dm-border) !important;
    background: transparent !important;
}

#main-tabs [role="tablist"] button {
    font-weight: 750 !important;
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


/* ---------- quick confusion-matrix guide ---------- */
.confusion-guide {
    background: var(--dm-surface) !important;
    border: 1px solid var(--dm-border) !important;
    border-radius: 14px;
    padding: 12px 14px;
    margin: 8px 0 10px;
    box-shadow: 0 6px 18px var(--dm-shadow);
}

.confusion-guide-title {
    font-weight: 850;
    font-size: .95rem;
    color: var(--dm-text) !important;
    margin-bottom: 9px;
}

.confusion-guide-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
}

.confusion-guide-item {
    border: 1px solid var(--dm-border);
    border-radius: 10px;
    padding: 9px 10px;
    background: var(--dm-surface-2);
}

.confusion-guide-item .metric {
    font-weight: 850;
    font-size: .84rem;
    margin-bottom: 3px;
}

.confusion-guide-item .meaning {
    font-size: .78rem;
    line-height: 1.35;
    color: var(--dm-text-muted) !important;
}

.confusion-guide-item.tp {
    border-left: 4px solid #16A34A;
}

.confusion-guide-item.fp {
    border-left: 4px solid #DC2626;
}

.confusion-guide-item.tn {
    border-left: 4px solid #2563EB;
}

.confusion-guide-item.fn {
    border-left: 4px solid #F59E0B;
}

@media (max-width: 800px) {
    .confusion-guide-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 520px) {
    .confusion-guide-grid {
        grid-template-columns: 1fr;
    }
}



/* ---------- focused participant questionnaire ---------- */
.dm-hero-compact {
    background: linear-gradient(135deg, #071A2D 0%, #6B007B 100%) !important;
    padding: 14px 18px !important;
    min-height: 0 !important;
    margin-bottom: 8px !important;
}

.dm-hero-compact .dm-hero-content {
    max-width: 100% !important;
}

.dm-hero-compact .dm-eyebrow {
    margin-bottom: 4px !important;
    font-size: .74rem !important;
}

.dm-hero-compact h1 {
    margin: 0 !important;
    font-size: clamp(1.3rem, 2vw, 1.8rem) !important;
    line-height: 1.15 !important;
}

.compact-lock {
    padding: 7px 11px !important;
    margin: 5px 0 7px !important;
}

.compact-lock .reference-title {
    margin: 0 !important;
    font-size: .88rem !important;
}



/* ---------- compact participant progress ---------- */
.compact-progress-card {
    padding: 10px 12px !important;
    margin: 6px 0 8px !important;
}

.compact-progress-row {
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: nowrap;
}

.compact-progress-title {
    font-weight: 850;
    font-size: .92rem;
    color: var(--dm-text) !important;
    margin-right: auto;
}

.compact-progress-item {
    display: flex;
    align-items: baseline;
    gap: 5px;
    white-space: nowrap;
}

.compact-progress-item strong {
    font-size: .95rem;
    color: var(--dm-text) !important;
}

.compact-progress-item span {
    font-size: .76rem;
    color: var(--dm-text-muted) !important;
}

.compact-progress-shell {
    margin-top: 7px !important;
}

@media (max-width: 560px) {
    .compact-progress-row {
        gap: 10px;
    }

    .compact-progress-title {
        font-size: .86rem;
    }

    .compact-progress-item strong {
        font-size: .88rem;
    }

    .compact-progress-item span {
        font-size: .72rem;
    }
}



/* ---------- compact collapsible study reference ---------- */
.reference-collapsible {
    padding: 0 !important;
    margin: 6px 0 8px !important;
    overflow: hidden;
}

.reference-collapsible summary {
    list-style: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 12px;
    user-select: none;
}

.reference-collapsible summary::-webkit-details-marker {
    display: none;
}

.reference-summary-label {
    font-size: .72rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--dm-text-muted) !important;
}

.reference-summary-state {
    font-size: .86rem;
    font-weight: 850;
    color: var(--dm-text) !important;
}

.reference-summary-toggle {
    margin-left: auto;
    font-size: .72rem;
    color: var(--dm-text-muted) !important;
}

.reference-collapsible[open] .reference-summary-toggle::after {
    content: "Hide details";
    font-size: 0;
}

.reference-collapsible[open] .reference-summary-toggle {
    font-size: 0;
}

.reference-collapsible[open] .reference-summary-toggle::before {
    content: "Hide details";
    font-size: .72rem;
}

.reference-body {
    border-top: 1px solid var(--dm-border);
    padding: 10px 12px 11px;
}

.compact-reference-details {
    display: flex !important;
    flex-wrap: wrap;
    gap: 6px 18px !important;
    margin: 0 !important;
}

.compact-reference-details span {
    font-size: .79rem !important;
}

.compact-reference-text {
    margin-top: 8px !important;
    font-size: .78rem !important;
    line-height: 1.4 !important;
}

@media (max-width: 650px) {
    .reference-collapsible summary {
        align-items: flex-start;
        flex-wrap: wrap;
    }

    .reference-summary-toggle {
        width: 100%;
        margin-left: 0;
    }

    .compact-reference-details {
        flex-direction: column;
        gap: 4px !important;
    }
}









/* ---------- Yes / No layout; selected colors are applied by JS ---------- */
#q1-map-indication .block-info,
#q2-right-reason .block-info,
#q1-map-indication [data-testid="block-info"],
#q2-right-reason [data-testid="block-info"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    border-radius: 0 !important;
    font-weight: 800 !important;
}

#q1-map-indication [role="radiogroup"],
#q2-right-reason [role="radiogroup"] {
    display: flex !important;
    gap: 10px !important;
    flex-wrap: nowrap !important;
}

#q1-map-indication [role="radiogroup"] label,
#q2-right-reason [role="radiogroup"] label {
    min-width: 90px !important;
    padding: 8px 13px !important;
    border-radius: 9px !important;
    border: 1px solid #D7DEE7 !important;
    background: #F8FAFC !important;
    color: #475569 !important;
    box-shadow: none !important;
    transition: opacity .12s ease, filter .12s ease, background .12s ease,
                border-color .12s ease, color .12s ease !important;
}


/* ---------- stronger theme-aware borders for Yes / No ---------- */

/* LIGHT THEME:
   Yes = thick green border
   No  = thick red border */
#q1-map-indication [role="radiogroup"] label:nth-child(1),
#q2-right-reason [role="radiogroup"] label:nth-child(1) {
    border: 3px solid #2FA84F !important;
    border-radius: 10px !important;
    background: #F3FBF5 !important;
}

#q1-map-indication [role="radiogroup"] label:nth-child(2),
#q2-right-reason [role="radiogroup"] label:nth-child(2) {
    border: 3px solid #D94A4A !important;
    border-radius: 10px !important;
    background: #FFF5F5 !important;
}

/* Give the radio circle itself a visible outline. */
#q1-map-indication input[type="radio"],
#q2-right-reason input[type="radio"] {
    width: 17px !important;
    height: 17px !important;
    outline: 2px solid rgba(71, 85, 105, .35) !important;
    outline-offset: 1px !important;
}

/* Dark theme: use clear white borders for both choices. */
.dark #q1-map-indication [role="radiogroup"] label,
.dark #q2-right-reason [role="radiogroup"] label {
    border: 2px solid rgba(255,255,255,.88) !important;
    background: rgba(255,255,255,.035) !important;
}

.dark #q1-map-indication input[type="radio"],
.dark #q2-right-reason input[type="radio"] {
    outline-color: rgba(255,255,255,.9) !important;
}

/* Respect system dark mode too. */
@media (prefers-color-scheme: dark) {
    #q1-map-indication [role="radiogroup"] label,
    #q2-right-reason [role="radiogroup"] label {
        border: 2px solid rgba(255,255,255,.88) !important;
        background: rgba(255,255,255,.035) !important;
    }

    #q1-map-indication input[type="radio"],
    #q2-right-reason input[type="radio"] {
        outline-color: rgba(255,255,255,.9) !important;
    }
}


/* ---------- focused evaluation container ---------- */
#evaluation-focus-card {
    border: 3px solid #A9B7C8 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    margin: 14px 0 18px !important;
    background: rgba(255,255,255,.72) !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .07) !important;
}

/* Keep the inner sections visually contained without extra outer clutter. */
#evaluation-focus-card > div {
    max-width: 100% !important;
}

/* Dark theme: clear white boundary around the full task area. */
.dark #evaluation-focus-card {
    border-color: rgba(255,255,255,.88) !important;
    background: rgba(17,24,39,.78) !important;
    box-shadow: 0 8px 26px rgba(0,0,0,.24) !important;
}

@media (prefers-color-scheme: dark) {
    #evaluation-focus-card {
        border-color: rgba(255,255,255,.88) !important;
        background: rgba(17,24,39,.78) !important;
        box-shadow: 0 8px 26px rgba(0,0,0,.24) !important;
    }
}

@media (max-width: 700px) {
    #evaluation-focus-card {
        padding: 12px !important;
        border-radius: 14px !important;
    }
}


/* ---------- self-explaining system specification ---------- */
.system-spec-card {
    border: 1px solid #D5DEE8;
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0 0 12px;
    background: #F8FAFC;
}

.system-spec-title {
    font-size: .95rem;
    font-weight: 850;
    color: #14213D;
    margin-bottom: 8px;
}

.system-spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.system-spec-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 9px 11px;
    border-radius: 9px;
    font-size: .82rem;
    line-height: 1.35;
}

.system-spec-item strong {
    font-size: .82rem;
}

.system-spec-item.normal {
    background: #EEF8F1;
    border-left: 4px solid #4FAF6A;
}

.system-spec-item.unexpected {
    background: #FCEFEF;
    border-left: 4px solid #D96B6B;
}

.video-role-guide {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: .78rem;
    color: #52606D;
    margin: 4px 0 7px;
    padding: 0 2px;
}

.video-role-guide strong {
    color: #14213D;
    margin-right: 3px;
}

.video-role-guide .divider {
    color: #94A3B8;
}

@media (max-width: 700px) {
    .system-spec-grid {
        grid-template-columns: 1fr;
    }

    .video-role-guide .divider {
        display: none;
    }
}



/* ---------- collapsible system specification ---------- */
.system-spec-collapsible {
    padding: 0 !important;
    overflow: hidden;
}

.system-spec-collapsible summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    user-select: none;
}

.system-spec-collapsible summary::-webkit-details-marker {
    display: none;
}

.system-spec-collapsible summary::after {
    content: "▾";
    margin-left: auto;
    color: #64748B;
    font-size: .9rem;
    transition: transform .15s ease;
}

.system-spec-collapsible[open] summary::after {
    transform: rotate(180deg);
}

.system-spec-collapsible .system-spec-title {
    margin: 0 !important;
}

.system-spec-toggle {
    color: #64748B !important;
    font-size: .76rem;
    margin-left: auto;
}

.system-spec-collapsible[open] .system-spec-toggle {
    font-size: 0;
}

.system-spec-collapsible[open] .system-spec-toggle::before {
    content: "Hide details";
    font-size: .76rem;
}

.system-spec-body {
    border-top: 1px solid #E2E8F0;
    padding: 10px 12px 12px;
}


/* ---------- three-part system specification ---------- */
.system-spec-grid-three {
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
}

.system-spec-item.false-positive {
    background: #FFF7E8;
    border-left: 4px solid #E59A2F;
}

@media (max-width: 900px) {
    .system-spec-grid-three {
        grid-template-columns: 1fr !important;
    }
}

/* ---------- surveyor help ---------- */
.surveyor-help {
    border: 1px solid #CBD5E1 !important;
    border-radius: 11px;
    background: #FFFFFF !important;
    margin: 0 0 12px !important;
    overflow: hidden;
}

.surveyor-help summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    user-select: none;
}

.surveyor-help summary::-webkit-details-marker {
    display: none;
}

.surveyor-help summary::after {
    content: "▾";
    margin-left: auto;
    color: #64748B;
    font-size: .9rem;
    transition: transform .15s ease;
}

.surveyor-help[open] summary::after {
    transform: rotate(180deg);
}

.surveyor-help-title {
    font-weight: 850;
    color: #14213D !important;
    font-size: .9rem;
}

.surveyor-help-subtitle {
    color: #64748B !important;
    font-size: .76rem;
}

.surveyor-help-body {
    border-top: 1px solid #E2E8F0;
    padding: 10px 12px 11px;
    display: grid;
    gap: 7px;
    font-size: .8rem;
    line-height: 1.4;
    color: #475569 !important;
}

.surveyor-help-step strong {
    color: #14213D !important;
}

.help-green {
    color: #16763A !important;
    font-weight: 800;
}

.help-red {
    color: #B52D2D !important;
    font-weight: 800;
}

@media (max-width: 700px) {
    .surveyor-help summary {
        align-items: flex-start;
        flex-wrap: wrap;
    }

    .surveyor-help-subtitle {
        width: 100%;
    }
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
#q2-right-reason [role="radiogroup"] label {
    cursor: pointer;
}

#q1-map-indication input[type="radio"],
#q2-right-reason input[type="radio"] {
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


/* ---------- Database Manager ---------- */
.database-hero {
    background: linear-gradient(135deg, #21002F 0%, #6B007B 55%, #9F01BE 100%);
}

.db-status-card {
    margin-top: 12px;
}

.danger-zone {
    background: #FFF1F2 !important;
    border: 1px solid #FDA4AF !important;
    border-left: 5px solid #DC2626 !important;
    border-radius: 14px;
    padding: 14px 16px;
    margin: 8px 0 14px;
}

.danger-zone-title {
    color: #991B1B !important;
    font-size: 1.05rem;
    font-weight: 850;
}

.danger-zone .study-meta {
    color: #7F1D1D !important;
}

.dark .danger-zone {
    background: #3A1017 !important;
    border-color: #7F1D1D !important;
}

.dark .danger-zone-title,
.dark .danger-zone .study-meta {
    color: #FECACA !important;
}

@media (prefers-color-scheme: dark) {
    .danger-zone {
        background: #3A1017 !important;
        border-color: #7F1D1D !important;
    }

    .danger-zone-title,
    .danger-zone .study-meta {
        color: #FECACA !important;
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




YES_NO_UI_JS = r"""
() => {
    const ids = ["q1-map-indication", "q2-right-reason"];

    function setImp(el, prop, value) {
        if (el) el.style.setProperty(prop, value, "important");
    }

    function clearImp(el, prop) {
        if (el) el.style.removeProperty(prop);
    }

    function styleGroup(id) {
        const root = document.getElementById(id);
        if (!root) return;

        const group = root.querySelector('[role="radiogroup"]') || root;
        const labels = Array.from(group.querySelectorAll("label"));
        if (!labels.length) return;

        const selectedLabel = labels.find(label => {
            const input = label.querySelector('input[type="radio"]');
            return input && input.checked;
        });

        labels.forEach((label, index) => {
            const input = label.querySelector('input[type="radio"]');
            const text = (label.innerText || label.textContent || "").trim().toLowerCase();
            const isYes = text === "yes" || text.endsWith(" yes");
            const isNo = text === "no" || text.endsWith(" no");
            const isSelected = selectedLabel === label;

            // Reset any Gradio blue state first.
            setImp(label, "box-shadow", "none");
            setImp(label, "transform", "none");

            if (!selectedLabel) {
                setImp(label, "opacity", "1");
                setImp(label, "filter", "none");
                setImp(label, "background", "#F8FAFC");
                setImp(label, "border-color", "#D7DEE7");
                setImp(label, "color", "#475569");
                if (input) setImp(input, "accent-color", "#94A3B8");
                return;
            }

            if (!isSelected) {
                // Opposite option: deliberately faint and neutral.
                setImp(label, "opacity", "0.18");
                setImp(label, "filter", "grayscale(1) saturate(0)");
                setImp(label, "background", "#F8FAFC");
                setImp(label, "border-color", "#E2E8F0");
                setImp(label, "color", "#94A3B8");
                if (input) setImp(input, "accent-color", "#CBD5E1");
                return;
            }

            // Selected answer.
            setImp(label, "opacity", "1");
            setImp(label, "filter", "none");
            setImp(label, "font-weight", "700");

            if (isYes || index === 0) {
                setImp(label, "background", "#BDE9C9");
                setImp(label, "border-width", "4px");
                setImp(label, "border-color", "#168A39");
                setImp(label, "color", "#0F4F25");
                if (input) setImp(input, "accent-color", "#168A39");
            } else if (isNo || index === 1) {
                setImp(label, "background", "#F3BBBB");
                setImp(label, "border-width", "4px");
                setImp(label, "border-color", "#B92525");
                setImp(label, "color", "#681717");
                if (input) setImp(input, "accent-color", "#B92525");
            }
        });
    }

    function refreshAll() {
        ids.forEach(styleGroup);
    }

    // Style immediately.
    refreshAll();

    // User selection changes.
    document.addEventListener("change", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLInputElement) || target.type !== "radio") return;
        const root = target.closest("#q1-map-indication, #q2-right-reason");
        if (!root) return;
        requestAnimationFrame(() => styleGroup(root.id));
        setTimeout(() => styleGroup(root.id), 60);
    }, true);

    // Gradio can re-render components after callbacks. Reapply styles.
    const observer = new MutationObserver(() => {
        requestAnimationFrame(refreshAll);
    });
    observer.observe(document.body, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ["class", "checked", "aria-checked"]
    });

    return [];
}
"""


DEFAULT_LIGHT_THEME_JS = r"""
() => {
    // Start the study in light mode regardless of the operating-system theme.
    document.documentElement.classList.remove("dark");
    document.body.classList.remove("dark");
    document.documentElement.style.colorScheme = "light";

    // Gradio/browser builds may consult one of these values.
    try {
        localStorage.setItem("theme", "light");
        localStorage.setItem("__theme", "light");
    } catch (e) {}

    // Some Gradio builds reapply the initial theme shortly after mount.
    requestAnimationFrame(() => {
        document.documentElement.classList.remove("dark");
        document.body.classList.remove("dark");
        document.documentElement.style.colorScheme = "light";
    });

    setTimeout(() => {
        document.documentElement.classList.remove("dark");
        document.body.classList.remove("dark");
        document.documentElement.style.colorScheme = "light";
    }, 100);

    return [];
}
"""


AUTOPLAY_NEXT_VIDEO_JS = r"""
() => {
    const root = document.getElementById("study-video");
    if (!root) return [];

    let attempts = 0;
    const maxAttempts = 30;

    const tryPlay = () => {
        attempts += 1;
        const video = root.querySelector("video");

        if (video) {
            // Reset to the beginning of the newly loaded trial.
            try {
                video.currentTime = 0;
            } catch (e) {}

            const promise = video.play();
            if (promise && typeof promise.catch === "function") {
                promise.catch(() => {
                    // Some browsers may briefly block play while the new source
                    // is still loading. Retry for a short period.
                    if (attempts < maxAttempts) {
                        setTimeout(tryPlay, 120);
                    }
                });
            }
            return;
        }

        if (attempts < maxAttempts) {
            setTimeout(tryPlay, 120);
        }
    };

    // Allow Gradio time to render/update the next video source first.
    setTimeout(tryPlay, 120);
    return [];
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
                """
                <section class="dm-hero dm-hero-compact">
                    <div class="dm-hero-content">
                        <div class="dm-eyebrow">DistriMuSe · XAI User Study</div>
                        <h1>Explainable Unexpected-Condition Detection Study</h1>
                    </div>
                </section>
                """
            )

            state = gr.State()

            with gr.Row():
                new_participant_button = gr.Button(
                    "Start study",
                    variant="primary",
                    scale=1,
                )

            # Internal participant identifier used for data storage only.
            participant_id = gr.Textbox(value="", visible=False)

            with gr.Group(elem_id="evaluation-focus-card"):
                gr.HTML(
                    """
                    <details class="surveyor-help">
                        <summary>
                            <span class="surveyor-help-title">What should a surveyor do?</span>
                            <span class="surveyor-help-subtitle">Learn how to fill this</span>
                        </summary>
                        <div class="surveyor-help-body">
                            <div class="surveyor-help-step">
                                <strong>1. Watch the full video.</strong>
                                Do not answer before playback reaches the end.
                            </div>
                            <div class="surveyor-help-step">
                                <strong>2. Check the left panel.</strong>
                                It contains the input video and the safety-area detections.
                                A <span class="help-green">green border</span> means that
                                safety area is considered normal.
                            </div>
                            <div class="surveyor-help-step">
                                <strong>3. Look for red detections.</strong>
                                A <span class="help-red">red border</span> indicates a
                                potential unexpected condition that should be verified
                                using the right panel.
                            </div>
                            <div class="surveyor-help-step">
                                <strong>4. Check the right panel.</strong>
                                It shows the system output / anomaly explanation. Use it
                                to decide whether the system behavior is correct and whether
                                the highlighted area explains the decision.
                            </div>
                        </div>
                    </details>
                    """
                )

                gr.HTML(
                    """
                    <details class="system-spec-card system-spec-collapsible">
                        <summary>
                            <span class="system-spec-title">What should the system do?</span>
                            <span class="system-spec-toggle">Show details</span>
                        </summary>
                        <div class="system-spec-body">
                            <div class="system-spec-grid system-spec-grid-three">
                                <div class="system-spec-item unexpected">
                                    <strong>Anomaly / Unexpected</strong>
                                    <span>
                                        Unauthorized persons, unsafe movements, faults,
                                        misplaced boxes, or other interference outside
                                        the normal palletizing process.
                                    </span>
                                </div>
                                <div class="system-spec-item normal">
                                    <strong>Normal</strong>
                                    <span>
                                        Machine palletizing, normal operator activity,
                                        pallet replacement, etc.
                                    </span>
                                </div>
                                <div class="system-spec-item false-positive">
                                    <strong>False Positive</strong>
                                    <span>
                                        A normal situation is incorrectly flagged as
                                        anomalous — a false alarm.
                                    </span>
                                </div>
                            </div>
                        </div>
                    </details>
                    """
                )

                status = gr.Markdown("")
                participant_progress = gr.HTML("")
                sequence_reference = gr.HTML("")

                gr.HTML(
                    """
                    <div class="video-role-guide">
                        <strong>Video</strong>
                        <span>Original scene</span>
                        <span class="divider">|</span>
                        <span>System output / anomaly explanation</span>
                    </div>
                    """
                )

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
                    map_indicates_unexpected = gr.Radio(
                        QUESTION_OPTIONS,
                        label=(
                            "1. Is the system behavior shown in this video consistent "
                            "with the description above?"
                        ),
                        interactive=True,
                        elem_id="q1-map-indication",
                    )
                    q1_error = gr.HTML("")

                    right_reason = gr.Radio(
                        QUESTION_OPTIONS,
                        label=(
                            "2. Does the highlighted area correctly explain the system decision?"
                        ),
                        interactive=True,
                        elem_id="q2-right-reason",
                    )
                    q2_error = gr.HTML("")

                submit_button = gr.Button(
                    "Submit & Next",
                    variant="primary",
                    visible=False,
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

                The questionnaire uses two direct checks:

                - **System behavior consistency**: whether the system behavior shown
                  in the video is coherent with the stated Normal/Unexpected specification.
                - **Explanation correctness**: whether the highlighted area correctly
                  explains the system decision.

                Each question uses **Yes or No**.
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


        with gr.Tab("Database Manager", id="database-manager"):
            gr.HTML(
                """
                <section class="researcher-hero database-hero">
                    <div class="dm-eyebrow">DistriMuSe · UC3 · Protected Administration</div>
                    <h2>Database Manager</h2>
                    <p>
                        This page contains study administration and destructive database
                        controls. Enter the researcher password to continue.
                    </p>
                </section>
                """
            )

            db_lock_status = gr.HTML(
                """
                <div class="db-locked-message">
                    🔒 Database Manager locked.
                </div>
                """
            )

            with gr.Row(elem_id="db-login-row"):
                db_password = gr.Textbox(
                    label="Database Manager password",
                    type="password",
                    placeholder="Enter researcher password",
                    scale=3,
                )
                db_unlock_button = gr.Button(
                    "Unlock Database Manager",
                    variant="primary",
                    scale=1,
                )
                db_lock_button = gr.Button(
                    "Lock",
                    variant="secondary",
                    scale=1,
                )

            with gr.Group(visible=False, elem_id="database-manager-content") as db_manager_content:
                gr.HTML(
                    """
                    <section class="researcher-hero database-hero">
                        <div class="dm-eyebrow">DistriMuSe · UC3 · Study Administration</div>
                        <h2>Database Manager</h2>
                        <p>
                            Inspect active participants and responses, export or back up the
                            SQLite database, archive individual records, restore archived
                            participants, and synchronize deletions with the configured
                            Hugging Face Dataset backend.
                        </p>
                    </section>
                    """
                )

                db_status = gr.HTML(database_status_html())

                with gr.Row():
                    db_refresh_button = gr.Button("Refresh database", variant="primary")
                    db_export_button = gr.Button("Export responses CSV")
                    db_backup_button = gr.Button("Backup SQLite database")

                with gr.Row():
                    db_export_file = gr.File(label="CSV export", interactive=False)
                    db_backup_file = gr.File(label="SQLite backup", interactive=False)

                gr.Markdown("### Active participants")
                db_participant_table = gr.Dataframe(
                    value=participant_records_dataframe(),
                    interactive=False,
                    wrap=True,
                    label="Participant records",
                )

                gr.Markdown("### Active responses")
                db_response_table = gr.Dataframe(
                    value=active_responses_dataframe(),
                    interactive=False,
                    wrap=True,
                    label="Response records",
                )

                gr.HTML('<div class="researcher-divider"></div>')

                gr.Markdown(
                    """
                    ### Archive or delete a record

                    Destructive actions create a timestamped SQLite backup first.
                    Deleted data is moved into archive tables, so it can be restored later.

                    If remote deletion is enabled and `HF_TOKEN` / `HF_DATA_REPO` are
                    configured, the corresponding rows are also removed from the remote
                    `responses.csv`.
                    """
                )

                db_participant_choices = participant_choices()
                db_selected_participant = gr.Dropdown(
                    choices=db_participant_choices,
                    value=db_participant_choices[0] if db_participant_choices else None,
                    label="Participant",
                )

                initial_response_choices = response_choices(
                    db_participant_choices[0] if db_participant_choices else None
                )
                db_selected_response = gr.Dropdown(
                    choices=initial_response_choices,
                    value=initial_response_choices[0][1] if initial_response_choices else None,
                    label="Response / video",
                )

                db_reason = gr.Textbox(
                    label="Deletion reason (optional)",
                    placeholder="e.g. test participant, duplicate response, invalid trial",
                )

                db_delete_remote = gr.Checkbox(
                    value=True,
                    label="Also delete matching response data from Hugging Face Dataset",
                )

                with gr.Row():
                    db_delete_confirmation = gr.Textbox(
                        label='Type "DELETE" to confirm',
                        placeholder="DELETE",
                    )
                    db_delete_response_button = gr.Button(
                        "Archive selected response",
                        variant="secondary",
                    )
                    db_delete_participant_button = gr.Button(
                        "Archive participant + responses",
                        variant="stop",
                    )

                db_action_status = gr.Markdown("")

                gr.HTML('<div class="researcher-divider"></div>')

                gr.Markdown("### Archived participants / restore")
                db_archived_table = gr.Dataframe(
                    value=archived_participants_dataframe(),
                    interactive=False,
                    wrap=True,
                    label="Archived participant records",
                )

                archived_choices = archived_participant_choices()
                db_restore_participant = gr.Dropdown(
                    choices=archived_choices,
                    value=archived_choices[0] if archived_choices else None,
                    label="Archived participant",
                )
                db_restore_confirmation = gr.Textbox(
                    label='Type "RESTORE" to confirm',
                    placeholder="RESTORE",
                )
                db_restore_button = gr.Button("Restore archived participant")
                db_restore_status = gr.Markdown("")

                gr.HTML('<div class="researcher-divider"></div>')

                gr.HTML(
                    """
                    <div class="danger-zone">
                        <div class="danger-zone-title">Danger Zone</div>
                        <div class="study-meta">
                            Resetting the study archives every active participant and response,
                            then clears the active study tables. A backup is created first.
                        </div>
                    </div>
                    """
                )

                db_reset_reason = gr.Textbox(
                    label="Reset reason (optional)",
                    placeholder="e.g. remove pilot data before official study",
                )
                db_reset_remote = gr.Checkbox(
                    value=True,
                    label="Also clear the remote Hugging Face responses.csv",
                )
                db_reset_confirmation = gr.Textbox(
                    label='Type "DELETE ALL" to confirm',
                    placeholder="DELETE ALL",
                )
                db_reset_button = gr.Button(
                    "Reset entire active study",
                    variant="stop",
                )
                db_reset_status = gr.Markdown("")

    gr.HTML(
        """
        <footer class="dm-footer" id="global-footer">
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

    demo.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=DEFAULT_LIGHT_THEME_JS,
    )

    demo.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=YES_NO_UI_JS,
    )

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
    ).then(
        fn=None,
        inputs=[],
        outputs=[],
        js=AUTOPLAY_NEXT_VIDEO_JS,
    )


    video_finished_trigger.click(
        unlock_questions_after_video,
        inputs=[state],
        outputs=[
            state,
            sequence_reference,
            questions_group,
            submit_button,
            map_indicates_unexpected,
            right_reason,
        ],
    )


    submit_button.click(
        submit_response,
        inputs=[
            state,
            map_indicates_unexpected,
            right_reason,
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
            map_indicates_unexpected,
            right_reason,
            questions_group,
            submit_button,
        ],
    ).then(
        fn=None,
        inputs=[],
        outputs=[],
        js=AUTOPLAY_NEXT_VIDEO_JS,
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

    db_selected_participant.change(
        update_response_dropdown,
        inputs=[db_selected_participant],
        outputs=[db_selected_response],
    )

    db_refresh_button.click(
        refresh_database_manager,
        outputs=[
            db_status,
            db_participant_table,
            db_response_table,
            db_archived_table,
            db_selected_participant,
            db_selected_response,
            db_restore_participant,
        ],
    )

    db_export_button.click(
        export_local_csv,
        outputs=[db_export_file],
    )

    db_backup_button.click(
        backup_database,
        outputs=[db_backup_file],
    )

    db_delete_response_button.click(
        archive_single_response,
        inputs=[
            db_selected_participant,
            db_selected_response,
            db_delete_confirmation,
            db_reason,
            db_delete_remote,
        ],
        outputs=[db_action_status],
    ).then(
        refresh_database_manager,
        outputs=[
            db_status,
            db_participant_table,
            db_response_table,
            db_archived_table,
            db_selected_participant,
            db_selected_response,
            db_restore_participant,
        ],
    )

    db_delete_participant_button.click(
        archive_participant,
        inputs=[
            db_selected_participant,
            db_delete_confirmation,
            db_reason,
            db_delete_remote,
        ],
        outputs=[db_action_status],
    ).then(
        refresh_database_manager,
        outputs=[
            db_status,
            db_participant_table,
            db_response_table,
            db_archived_table,
            db_selected_participant,
            db_selected_response,
            db_restore_participant,
        ],
    )

    db_restore_button.click(
        restore_archived_participant,
        inputs=[
            db_restore_participant,
            db_restore_confirmation,
        ],
        outputs=[db_restore_status],
    ).then(
        refresh_database_manager,
        outputs=[
            db_status,
            db_participant_table,
            db_response_table,
            db_archived_table,
            db_selected_participant,
            db_selected_response,
            db_restore_participant,
        ],
    )

    db_reset_button.click(
        reset_entire_study,
        inputs=[
            db_reset_confirmation,
            db_reset_reason,
            db_reset_remote,
        ],
        outputs=[db_reset_status],
    ).then(
        refresh_database_manager,
        outputs=[
            db_status,
            db_participant_table,
            db_response_table,
            db_archived_table,
            db_selected_participant,
            db_selected_response,
            db_restore_participant,
        ],
    )


    db_unlock_button.click(
        unlock_database_manager,
        inputs=[db_password],
        outputs=[
            db_lock_status,
            db_manager_content,
            db_password,
        ],
    )

    db_password.submit(
        unlock_database_manager,
        inputs=[db_password],
        outputs=[
            db_lock_status,
            db_manager_content,
            db_password,
        ],
    )

    db_lock_button.click(
        lock_database_manager,
        outputs=[
            db_lock_status,
            db_manager_content,
            db_password,
        ],
    )


if __name__ == "__main__":
    demo.launch()
