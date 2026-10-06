import json
import shutil
import threading
from pathlib import Path
from typing import Any

from app.config import ACTIVE_JOBS, STORAGE
from app.db import ScanDatabase
from app.models import new_record


class ScanStore:
    def __init__(self, root: Path = STORAGE) -> None:
        self.root = root
        self.records = root / "scans"
        self.uploads = root / "uploads"
        self.processing = root / "processing"
        self.models = root / "models"
        for path in (self.records, self.uploads, self.processing, self.models):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: set[str] = set()
        self.db = ScanDatabase(root / "roomscan.db")
        self.db.replace_all(self.list_scans())

    def create(self, scan_id: str, name: str) -> dict[str, Any]:
        record = new_record(scan_id, name)
        with self._lock:
            self._write(record)
        return record

    def get(self, scan_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read(scan_id)

    def list_scans(self) -> list[dict[str, Any]]:
        with self._lock:
            records = []
            for path in self.records.glob("*.json"):
                records.append(json.loads(path.read_text(encoding="utf-8")))
        records.sort(key=lambda item: item["createdAt"], reverse=True)
        return records

    def update(self, scan_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self._require(scan_id)
            record.update(fields)
            self._write(record)
            return record

    def mark_interrupted(self) -> None:
        with self._lock:
            for path in self.records.glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                active = record.get("status") == "processing" or record.get("jobStatus") in ACTIVE_JOBS
                if active and record.get("status") != "completed":
                    record["status"] = "failed"
                    record["jobStatus"] = "FAILED"
                    record["errorMessage"] = "Reconstruction was interrupted. Try again."
                    record["estimatedSecondsRemaining"] = 0
                    self._write(record)

    def try_begin(self, scan_id: str) -> dict[str, Any] | None:
        """Reserve a processing job. Returns None when a job is already running or the model exists."""
        with self._lock:
            record = self._require(scan_id)
            if scan_id in self._jobs or record.get("status") == "processing":
                return None
            if record.get("hasModel") and record.get("status") == "completed":
                return None
            record.update(
                {
                    "status": "processing",
                    "progress": 8,
                    "processingStage": "analyzing",
                    "estimatedSecondsRemaining": 180,
                    "errorMessage": None,
                    "completedAt": None,
                    "hasModel": False,
                }
            )
            self._write(record)
            self._jobs.add(scan_id)
            return record

    def finish_job(self, scan_id: str) -> None:
        with self._lock:
            self._jobs.discard(scan_id)

    def is_running(self, scan_id: str) -> bool:
        with self._lock:
            return scan_id in self._jobs

    def begin_video_job(self, scan_id: str) -> dict[str, Any] | None:
        """Reserve a video job without changing the photo reconstruction lock rules."""
        with self._lock:
            if scan_id in self._jobs:
                return None
            record = self._require(scan_id)
            if record.get("hasModel") and record.get("status") == "completed":
                return None
            self._jobs.add(scan_id)
            return record

    def upload_dir(self, scan_id: str) -> Path:
        path = self.uploads / scan_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear_uploads(self, scan_id: str) -> Path:
        path = self.uploads / scan_id
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, scan_id: str) -> Path:
        path = self.processing / scan_id
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_path(self, scan_id: str) -> Path:
        return self.models / f"{scan_id}.glb"

    def workspace(self, scan_id: str) -> Path:
        path = self.records / scan_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def original_dir(self, scan_id: str) -> Path:
        path = self.workspace(scan_id) / "original"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def images_dir(self, scan_id: str) -> Path:
        path = self.workspace(scan_id) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def frames_path(self, scan_id: str) -> Path:
        return self.records / scan_id / "images"

    def colmap_dir(self, scan_id: str) -> Path:
        path = self.workspace(scan_id) / "colmap"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def openmvs_dir(self, scan_id: str) -> Path:
        path = self.workspace(scan_id) / "openmvs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output_dir(self, scan_id: str) -> Path:
        path = self.workspace(scan_id) / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output_glb(self, scan_id: str) -> Path:
        return self.records / scan_id / "output" / "room.glb"

    def host_workspace(self, scan_id: str) -> Path:
        from app.config import SCANS_HOST_PATH

        if SCANS_HOST_PATH:
            return Path(SCANS_HOST_PATH) / scan_id
        return self.workspace(scan_id)

    def clear_model(self, scan_id: str) -> None:
        path = self.model_path(scan_id)
        if path.exists():
            path.unlink()

    def _require(self, scan_id: str) -> dict[str, Any]:
        record = self._read(scan_id)
        if record is None:
            raise KeyError(scan_id)
        return record

    def _read(self, scan_id: str) -> dict[str, Any] | None:
        path = self.records / f"{scan_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, record: dict[str, Any]) -> None:
        path = self.records / f"{record['id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        self.db.upsert(record)
