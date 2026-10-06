import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class ScanDatabase:
    """SQLite mirror of scan records. JSON files remain the source of truth."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        self._conn.executescript(schema)
        self._lock = threading.Lock()

    def upsert(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO scans (
                    id, name, source, status, job_status, progress, processing_stage,
                    photo_count, frame_count, usable_frame_count, duration_seconds,
                    video_filename, video_bytes, extraction_fps, has_model,
                    error_message, created_at, completed_at, payload
                ) VALUES (
                    :id, :name, :source, :status, :job_status, :progress, :processing_stage,
                    :photo_count, :frame_count, :usable_frame_count, :duration_seconds,
                    :video_filename, :video_bytes, :extraction_fps, :has_model,
                    :error_message, :created_at, :completed_at, :payload
                )
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    source = excluded.source,
                    status = excluded.status,
                    job_status = excluded.job_status,
                    progress = excluded.progress,
                    processing_stage = excluded.processing_stage,
                    photo_count = excluded.photo_count,
                    frame_count = excluded.frame_count,
                    usable_frame_count = excluded.usable_frame_count,
                    duration_seconds = excluded.duration_seconds,
                    video_filename = excluded.video_filename,
                    video_bytes = excluded.video_bytes,
                    extraction_fps = excluded.extraction_fps,
                    has_model = excluded.has_model,
                    error_message = excluded.error_message,
                    created_at = excluded.created_at,
                    completed_at = excluded.completed_at,
                    payload = excluded.payload
                """,
                _row(record, payload),
            )
            self._conn.commit()

    def replace_all(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM scans")
            for record in records:
                payload = json.dumps(record)
                self._conn.execute(
                    """
                    INSERT INTO scans (
                        id, name, source, status, job_status, progress, processing_stage,
                        photo_count, frame_count, usable_frame_count, duration_seconds,
                        video_filename, video_bytes, extraction_fps, has_model,
                        error_message, created_at, completed_at, payload
                    ) VALUES (
                        :id, :name, :source, :status, :job_status, :progress, :processing_stage,
                        :photo_count, :frame_count, :usable_frame_count, :duration_seconds,
                        :video_filename, :video_bytes, :extraction_fps, :has_model,
                        :error_message, :created_at, :completed_at, :payload
                    )
                    """,
                    _row(record, payload),
                )
            self._conn.commit()


def _row(record: dict[str, Any], payload: str) -> dict[str, Any]:
    return {
        "id": record["id"],
        "name": record.get("name") or "New room",
        "source": record.get("source") or "photos",
        "status": record.get("status") or "capturing",
        "job_status": record.get("jobStatus"),
        "progress": record.get("progress"),
        "processing_stage": record.get("processingStage"),
        "photo_count": int(record.get("photoCount") or 0),
        "frame_count": record.get("frameCount"),
        "usable_frame_count": record.get("usableFrameCount"),
        "duration_seconds": record.get("durationSeconds"),
        "video_filename": record.get("videoFilename"),
        "video_bytes": record.get("videoBytes"),
        "extraction_fps": record.get("extractionFps"),
        "has_model": 1 if record.get("hasModel") else 0,
        "error_message": record.get("errorMessage"),
        "created_at": record.get("createdAt") or "",
        "completed_at": record.get("completedAt"),
        "payload": payload,
    }
