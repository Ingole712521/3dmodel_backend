import threading
from pathlib import Path

from app.config import MIN_PHOTOS
from app.models import utc_now
from app.services.colmap import ReconstructionError, reconstruct_points
from app.services.mesh import mesh_from_points
from app.services.storage import ScanStore

STAGE_PROGRESS = {
    "analyzing": (18, 150),
    "cameras": (42, 100),
    "point-cloud": (68, 70),
    "mesh": (82, 40),
    "textures": (93, 15),
    "done": (100, 0),
}


def start_reconstruction(store: ScanStore, scan_id: str) -> dict:
    record = store.get(scan_id)
    if record is None:
        raise KeyError(scan_id)
    photo_count = int(record.get("photoCount") or 0)
    if photo_count < MIN_PHOTOS:
        store.update(
            scan_id,
            status="failed",
            progress=0,
            errorMessage=f"Add at least {MIN_PHOTOS} photos before creating a model.",
            estimatedSecondsRemaining=0,
        )
        raise ReconstructionError(f"Add at least {MIN_PHOTOS} photos before creating a model.")

    reserved = store.try_begin(scan_id)
    if reserved is None:
        current = store.get(scan_id)
        if current is None:
            raise KeyError(scan_id)
        return current

    store.clear_model(scan_id)
    thread = threading.Thread(target=_run, args=(store, scan_id), name=f"reconstruct-{scan_id}", daemon=True)
    thread.start()
    latest = store.get(scan_id)
    if latest is None:
        raise KeyError(scan_id)
    return latest


def _run(store: ScanStore, scan_id: str) -> None:
    try:
        images = store.upload_dir(scan_id)
        work = store.work_dir(scan_id)
        _stage(store, scan_id, "analyzing")
        ply_path = reconstruct_points(images, work, lambda stage: _stage(store, scan_id, stage))
        _stage(store, scan_id, "mesh")
        model_path = store.model_path(scan_id)
        mesh_from_points(ply_path, model_path)
        _stage(store, scan_id, "textures")
        if not model_path.exists() or model_path.stat().st_size < 32:
            raise ReconstructionError("The mesh export did not produce a GLB file.")
        store.update(
            scan_id,
            status="completed",
            progress=100,
            processingStage="done",
            estimatedSecondsRemaining=0,
            completedAt=utc_now(),
            hasModel=True,
            errorMessage=None,
        )
    except ReconstructionError as exc:
        _fail(store, scan_id, str(exc))
    except Exception as exc:
        _fail(store, scan_id, f"Reconstruction failed: {exc}")
    finally:
        store.finish_job(scan_id)


def _stage(store: ScanStore, scan_id: str, stage: str) -> None:
    progress, remaining = STAGE_PROGRESS.get(stage, (50, 60))
    store.update(
        scan_id,
        status="processing",
        processingStage=stage,
        progress=progress,
        estimatedSecondsRemaining=remaining,
        errorMessage=None,
    )


def _fail(store: ScanStore, scan_id: str, message: str) -> None:
    store.clear_model(scan_id)
    store.update(
        scan_id,
        status="failed",
        progress=0,
        estimatedSecondsRemaining=0,
        hasModel=False,
        errorMessage=message.strip()[:1200],
    )


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
