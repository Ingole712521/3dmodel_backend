import json
import shutil
import threading
from pathlib import Path
from typing import Any

from app.config import STORAGE
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
                if record.get("status") == "processing":
                    record["status"] = "failed"
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
