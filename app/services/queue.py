import queue
import threading

from app.config import JOB_COMPLETED, JOB_EXPORTING, JOB_EXTRACTING, JOB_MESH, JOB_RECONSTRUCTING
from app.services.storage import ScanStore
from app.services.video_job import run_video_job

_jobs: queue.Queue[str] = queue.Queue()
_started = False
_start_lock = threading.Lock()
_store: ScanStore | None = None


def start_worker(store: ScanStore) -> None:
    global _started, _store
    with _start_lock:
        _store = store
        if _started:
            return
        _started = True
        thread = threading.Thread(target=_loop, name="scan-worker", daemon=True)
        thread.start()


def enqueue_video(store: ScanStore, scan_id: str) -> dict:
    record = store.get(scan_id)
    if record is None:
        raise KeyError(scan_id)
    if store.is_running(scan_id) or record.get("jobStatus") in {JOB_EXTRACTING, JOB_RECONSTRUCTING, JOB_MESH, JOB_EXPORTING}:
        return record
    if record.get("status") == "completed" and record.get("hasModel"):
        return record
    updated = store.update(
        scan_id,
        status="processing",
        jobStatus=JOB_EXTRACTING,
        processingStage="extracting",
        progress=12,
        estimatedSecondsRemaining=700,
        errorMessage=None,
        hasModel=False,
        completedAt=None,
    )
    start_worker(store)
    _jobs.put(scan_id)
    return updated


def _loop() -> None:
    while True:
        scan_id = _jobs.get()
        store = _store
        try:
            if store is not None:
                run_video_job(store, scan_id)
        finally:
            _jobs.task_done()


def job_status_of(record: dict) -> str:
    status = record.get("jobStatus")
    if isinstance(status, str) and status:
        return status
    if record.get("status") == "completed":
        return JOB_COMPLETED
    if record.get("status") == "failed":
        return "FAILED"
    return "UPLOADING"
