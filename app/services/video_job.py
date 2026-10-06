"""Video jobs extract frames, then build a virtual tour instead of a mesh."""

import shutil
from pathlib import Path

from app.config import (
    FRAME_FPS,
    JOB_COMPLETED,
    JOB_EXPORTING,
    JOB_EXTRACTING,
    JOB_FAILED,
    JOB_MESH,
    JOB_RECONSTRUCTING,
    MIN_PHOTOS,
    TARGET_MIN_FRAMES,
)
from app.models import utc_now
from app.services.colmap import ReconstructionError
from app.services.ffmpeg import extract_frames, probe_duration
from app.services.frames import filter_frames, refresh_valid_frames, validate_frames
from app.services.reconstruction_log import append_log, load_status, save_status, write_file_manifest
from app.services.storage import ScanStore
from app.services.tour import build_tour

_STAGE_FOR_JOB = {
    JOB_EXTRACTING: ("extracting", 18, 640),
    JOB_RECONSTRUCTING: ("reconstructing", 46, 480),
    JOB_MESH: ("mesh", 78, 160),
    JOB_EXPORTING: ("exporting", 92, 40),
    JOB_COMPLETED: ("done", 100, 0),
}


def run_video_job(store: ScanStore, scan_id: str) -> None:
    reserved = store.begin_video_job(scan_id)
    if reserved is None:
        return
    try:
        _execute(store, scan_id)
    except ReconstructionError as exc:
        _fail(store, scan_id, str(exc))
    except Exception as exc:
        _fail(store, scan_id, f"The virtual tour could not be created: {exc}")
    finally:
        store.finish_job(scan_id)


def _execute(store: ScanStore, scan_id: str) -> None:
    record = store.get(scan_id)
    if record is None:
        raise KeyError(scan_id)
    video = _video_path(store, scan_id, record.get("videoFilename"))
    if video is None:
        raise ReconstructionError("The uploaded video could not be found.")

    _set_job(store, scan_id, JOB_EXTRACTING)
    duration = float(record.get("durationSeconds") or 0)
    if duration <= 0:
        duration = probe_duration(video)
        store.update(scan_id, durationSeconds=duration)

    images = store.images_dir(scan_id)
    fps = float(record.get("extractionFps") or FRAME_FPS)
    raw_count, usable, fps = _extract_usable_frames(video, images, fps, duration)
    store.update(
        scan_id,
        frameCount=raw_count,
        usableFrameCount=usable,
        photoCount=usable,
        extractionFps=fps,
    )

    store.clear_model(scan_id)
    output = _fresh(store.output_dir(scan_id))
    scan_dir = store.host_workspace(scan_id)
    build_tour(images, output, scan_dir, lambda stage: _on_tour(store, scan_id, stage))
    tour_path = output / "tour.json"
    if not tour_path.exists() or tour_path.stat().st_size < 32:
        raise ReconstructionError("The tour export did not produce a tour.json file.")
    store.update(
        scan_id,
        status="completed",
        jobStatus=JOB_COMPLETED,
        progress=100,
        processingStage="done",
        estimatedSecondsRemaining=0,
        completedAt=utc_now(),
        hasModel=True,
        errorMessage=None,
    )


def _on_tour(store: ScanStore, scan_id: str, stage: str) -> None:
    job_status = {
        "keyframes": JOB_RECONSTRUCTING,
        "nodes": JOB_MESH,
        "graph": JOB_EXPORTING,
        "tour": JOB_EXPORTING,
    }.get(stage, JOB_RECONSTRUCTING)
    _set_job(store, scan_id, job_status)


def _set_job(store: ScanStore, scan_id: str, job_status: str) -> None:
    stage, progress, remaining = _STAGE_FOR_JOB[job_status]
    store.update(
        scan_id,
        status="processing",
        jobStatus=job_status,
        processingStage=stage,
        progress=progress,
        estimatedSecondsRemaining=remaining,
        errorMessage=None,
    )


def _fail(store: ScanStore, scan_id: str, message: str) -> None:
    scan_dir = store.host_workspace(scan_id)
    append_log(scan_dir, f"JOB FAILED: {message}")
    try:
        status = load_status(scan_dir)
        status["output_files"] = write_file_manifest(scan_dir)
        save_status(scan_dir, status)
    except OSError:
        pass
    store.clear_model(scan_id)
    glb = store.output_glb(scan_id)
    if glb.exists() and glb.stat().st_size < 32:
        glb.unlink()
    store.update(
        scan_id,
        status="failed",
        jobStatus=JOB_FAILED,
        progress=0,
        estimatedSecondsRemaining=0,
        hasModel=False,
        errorMessage=message.strip()[:1200],
    )


def _extract_usable_frames(video: Path, images: Path, fps: float, duration: float) -> tuple[int, int, float]:
    raw_count = extract_frames(video, images, fps)
    _prepare_frames(images)
    usable = _filter_or_corrupt(images)
    refresh_valid_frames(images, images.parent)
    if usable >= TARGET_MIN_FRAMES or duration <= 0:
        return raw_count, usable, fps

    higher = min(8.0, max(fps * 2, TARGET_MIN_FRAMES / duration))
    if higher <= fps + 0.4:
        return raw_count, usable, fps

    snapshot = images.parent / "_frames_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(images, snapshot)
    report_snapshot = images.parent / "_frame_reports"
    if report_snapshot.exists():
        shutil.rmtree(report_snapshot)
    shutil.copytree(images.parent / "logs", report_snapshot)
    try:
        second_raw = extract_frames(video, images, higher)
        _prepare_frames(images)
        second_usable = _filter_or_corrupt(images)
        refresh_valid_frames(images, images.parent)
    except ReconstructionError:
        if snapshot.exists():
            _restore_frames(snapshot, images)
            _restore_reports(report_snapshot, images.parent / "logs")
        return raw_count, usable, fps
    if second_usable < usable:
        _restore_frames(snapshot, images)
        _restore_reports(report_snapshot, images.parent / "logs")
        return raw_count, usable, fps
    shutil.rmtree(snapshot, ignore_errors=True)
    shutil.rmtree(report_snapshot, ignore_errors=True)
    return second_raw, second_usable, higher


def _prepare_frames(images: Path) -> None:
    validate_frames(images, images.parent)


def _filter_or_corrupt(images: Path) -> int:
    try:
        return filter_frames(images, MIN_PHOTOS)
    except ReconstructionError as exc:
        if "No frames were extracted" not in str(exc):
            raise
        invalid = images.parent / "logs" / "invalid_frames.txt"
        if invalid.exists():
            first = next((line.split()[0] for line in invalid.read_text(encoding="utf-8").splitlines() if line.strip()), "")
            if first:
                raise ReconstructionError(f"Corrupted frame detected: {first}") from exc
        raise


def _restore_reports(snapshot: Path, logs: Path) -> None:
    if not snapshot.exists():
        return
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("valid_frames.txt", "invalid_frames.txt"):
        source = snapshot / name
        if source.exists():
            shutil.copy2(source, logs / name)
    shutil.rmtree(snapshot, ignore_errors=True)


def _restore_frames(snapshot: Path, images: Path) -> None:
    if images.exists():
        shutil.rmtree(images)
    snapshot.rename(images)


def _video_path(store: ScanStore, scan_id: str, filename: str | None) -> Path | None:
    folder = store.original_dir(scan_id)
    if filename:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    videos = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".m4v"}]
    return videos[0] if videos else None


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
