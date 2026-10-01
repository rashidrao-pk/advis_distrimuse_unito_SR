
import csv
import hashlib
import io
import os
import random
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
VIDEO_DIR = APP_DIR / "videos"
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

METADATA_CSV = APP_DIR / "videos.csv"
SQLITE_PATH = DATA_DIR / "study.db"

# Optional Hugging Face persistent backend.
# If HF_TOKEN and HF_DATA_REPO are configured, every response is also
# synchronized to responses.csv in a Hugging Face Dataset repository.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_DATA_REPO = os.getenv("HF_DATA_REPO", "").strip()
HF_RESPONSES_FILE = "responses.csv"

DB_LOCK = threading.Lock()
HF_LOCK = threading.Lock()

QUESTION_OPTIONS = ["Yes", "Partially", "No", "Cannot determine"]

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


def init_db():
    with sqlite3.connect(SQLITE_PATH) as conn:
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


def trial_display(state):
    if not state or not state.get("remaining"):
        return (
            None,
            "### Study complete\nThank you. All available clips have been rated.",
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
        video_path = None
    else:
        status = (
            f"### Sequence {trial_number} / {total}\n"
            "Please watch the complete clip before answering."
        )

    return (
        video_path,
        status,
        gr.update(visible=True),
        gr.update(visible=True),
    )


def start_study(participant_id):
    pid = safe_participant_id(participant_id)
    if not pid:
        return (
            None,
            None,
            "Please enter a participant ID.",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    try:
        state = make_trial_state(pid)
        video, status, q_vis, submit_vis = trial_display(state)
        if not state["remaining"]:
            return (
                state,
                None,
                status,
                gr.update(visible=False),
                gr.update(visible=False),
            )
        return state, video, status, q_vis, submit_vis
    except Exception as exc:
        return (
            None,
            None,
            f"Could not start study: {exc}",
            gr.update(visible=False),
            gr.update(visible=False),
        )


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
            None,
            "No active trial.",
            None, None, None, None, None, "",
            gr.update(visible=False),
            gr.update(visible=False),
        )

    required = {
        "Does the map indicate an unexpected condition?": map_indicates_unexpected,
        "Right reason": right_reason,
        "Right area": right_area,
        "Localization score": localization_score,
        "Confidence": confidence,
    }
    missing = [k for k, v in required.items() if v in (None, "", [])]
    if missing:
        return (
            state,
            state["remaining"][0]["_path"],
            "Please answer all required questions before continuing.",
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
        "map_indicates_unexpected": map_indicates_unexpected,
        "right_reason": right_reason,
        "right_area": right_area,
        "localization_score": int(localization_score),
        "confidence": int(confidence),
        "comment": (comment or "").strip(),
        "response_seconds": round(response_seconds, 3),
        "submitted_at_utc": now.isoformat(),
    }

    synced, sync_error = insert_response(payload)

    state["remaining"] = state["remaining"][1:]
    state["started_at"] = datetime.now(timezone.utc).timestamp()

    video, status, q_vis, submit_vis = trial_display(state)

    if synced and HF_TOKEN and HF_DATA_REPO:
        sync_note = "\n\n✅ Response saved locally and synchronized."
    elif HF_TOKEN and HF_DATA_REPO:
        sync_note = (
            "\n\n⚠️ Response saved locally, but remote synchronization failed: "
            f"`{sync_error}`"
        )
    else:
        sync_note = "\n\n✅ Response saved."

    if state["remaining"]:
        status += sync_note
    else:
        status += sync_note

    return (
        state,
        video,
        status,
        None,
        None,
        None,
        None,
        None,
        "",
        q_vis,
        submit_vis,
    )


def export_local_csv():
    df = local_response_dataframe()
    path = DATA_DIR / "responses_export.csv"
    df.to_csv(path, index=False)
    return str(path)


init_db()

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#study-title { text-align: center; }
#instructions { font-size: 0.98rem; }
"""

with gr.Blocks(title="Explainable Anomaly Detection User Study", css=CSS) as demo:
    gr.Markdown(
        """
        # Explainable Unexpected-Condition Detection Study
        """,
        elem_id="study-title",
    )

    gr.Markdown(
        """
        **Purpose.** This study evaluates whether the anomaly/explanation map
        highlights an unexpected condition for the **right reason** and in the
        **right safety area**.

        Please judge the explanation shown in the video, not merely whether an
        anomaly detector produced an alert. Your responses are stored under the
        participant ID you enter below.
        """,
        elem_id="instructions",
    )

    state = gr.State()

    with gr.Row():
        participant_id = gr.Textbox(
            label="Participant ID",
            placeholder="Example: P001",
            scale=3,
        )
        start_button = gr.Button("Start / Resume", variant="primary", scale=1)

    status = gr.Markdown("Enter your participant ID to begin.")

    video = gr.Video(
        label="Study clip",
        interactive=False,
        height=560,
    )

    with gr.Group(visible=False) as questions_group:
        map_indicates_unexpected = gr.Radio(
            QUESTION_OPTIONS,
            label="1. Does the explanation map indicate an unexpected condition?",
        )

        right_reason = gr.Radio(
            QUESTION_OPTIONS,
            label=(
                "2. Does the highlighted region correspond to the actual cause "
                "of the unexpected condition?"
            ),
        )

        right_area = gr.Radio(
            QUESTION_OPTIONS,
            label=(
                "3. Is the unexpected condition highlighted in the correct "
                "safety area?"
            ),
        )

        localization_score = gr.Slider(
            1, 5, step=1, value=None,
            label=(
                "4. How well does the anomaly map localize the actual cause? "
                "(1 = very poor, 5 = excellent)"
            ),
        )

        confidence = gr.Slider(
            1, 5, step=1, value=None,
            label=(
                "5. How confident are you in your assessment? "
                "(1 = very low, 5 = very high)"
            ),
        )

        comment = gr.Textbox(
            label="Optional comment",
            lines=3,
            placeholder="Optional: explain what looks correct, incorrect, or ambiguous.",
        )

    submit_button = gr.Button(
        "Submit & Next",
        variant="primary",
        visible=False,
    )

    with gr.Accordion("Researcher tools", open=False):
        gr.Markdown(
            """
            Use this only on the researcher's local/admin copy of the app.
            It exports the locally stored responses to CSV.
            """
        )
        export_button = gr.Button("Export local responses")
        export_file = gr.File(label="CSV export")

    start_button.click(
        start_study,
        inputs=[participant_id],
        outputs=[
            state,
            video,
            status,
            questions_group,
            submit_button,
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

    export_button.click(
        export_local_csv,
        outputs=export_file,
    )

if __name__ == "__main__":
    demo.launch()
