from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import (
    JOB_FAILED,
    JOB_UPLOADING,
    MAX_VIDEO_BYTES,
    MAX_VIDEO_SECONDS,
    MIN_VIDEO_SECONDS,
    RECOMMENDED_MAX_SECONDS,
    RECOMMENDED_MIN_SECONDS,
)
from app.schemas import JobStatusResponse, VideoUploadResponse
from app.services.colmap import ReconstructionError
from app.services.ffmpeg import probe_duration
from app.services.queue import enqueue_video, job_status_of
from app.services.reconstruction import image_files
from app.services.storage import ScanStore
from app.utils.files import is_video_container, video_extension

router = APIRouter()


def _store(request: Request) -> ScanStore:
    return request.app.state.store


def _parse_id(scan_id: str) -> str:
    try:
        return str(UUID(scan_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="This scan could not be found.") from exc


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.post("/scan/video", response_model=VideoUploadResponse)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    scanId: str | None = Form(default=None),
    name: str = Form(default="New room"),
) -> VideoUploadResponse:
    store = _store(request)
    if scanId:
        parsed = _parse_id(scanId)
        record = store.get(parsed)
        if record is None:
            raise HTTPException(status_code=404, detail="This scan could not be found.")
        if store.is_running(parsed):
            raise HTTPException(status_code=409, detail="This scan is already being reconstructed.")
    else:
        parsed = str(uuid4())
        record = store.create(parsed, name)

    try:
        extension = video_extension(file.filename or "video.mp4")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.update(
        parsed,
        name=(name or record.get("name") or "New room").strip() or "New room",
        source="video",
        status="processing",
        jobStatus=JOB_UPLOADING,
        processingStage="uploading",
        progress=4,
        estimatedSecondsRemaining=720,
        errorMessage=None,
        hasModel=False,
        videoFilename=f"video{extension}",
    )

    destination = store.original_dir(parsed) / f"video{extension}"
    size = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video is larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB.",
                    )
                handle.write(chunk)
        if size < 32 or not is_video_container(destination):
            raise HTTPException(status_code=400, detail="Upload an MP4, MOV, or M4V video.")
        try:
            duration = probe_duration(destination)
        except ReconstructionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if duration < MIN_VIDEO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"This video is {duration:.0f} seconds. Record at least {MIN_VIDEO_SECONDS:.0f} seconds while walking through the room.",
            )
        if duration > MAX_VIDEO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"This video is {duration:.0f} seconds. Keep the walkthrough under {MAX_VIDEO_SECONDS / 60:.0f} minutes.",
            )
    except HTTPException as exc:
        destination.unlink(missing_ok=True)
        store.update(
            parsed,
            status="failed",
            jobStatus=JOB_FAILED,
            progress=0,
            estimatedSecondsRemaining=0,
            hasModel=False,
            errorMessage=str(exc.detail),
        )
        raise

    warning = None
    if duration < RECOMMENDED_MIN_SECONDS or duration > RECOMMENDED_MAX_SECONDS:
        warning = "A 60–90 second walkthrough gives the reconstruction a more complete view of the room."

    store.update(
        parsed,
        source="video",
        videoFilename=destination.name,
        videoBytes=size,
        durationSeconds=duration,
        extractionFps=None,
        warning=warning,
        frameCount=None,
        usableFrameCount=None,
        photoCount=0,
    )
    try:
        started = enqueue_video(store, parsed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="This scan could not be found.") from exc
    return VideoUploadResponse(
        scanId=parsed,
        status=job_status_of(started),
        durationSeconds=duration,
        videoBytes=size,
        warning=warning,
    )


@router.get("/scan/{scan_id}/status", response_model=JobStatusResponse)
def video_status(scan_id: str, request: Request) -> JobStatusResponse:
    record = _require(request, scan_id)
    return _job(record, _base(request))


@router.get("/scan/{scan_id}/model")
def video_model(scan_id: str, request: Request) -> FileResponse:
    parsed = _parse_id(scan_id)
    record = _require(request, parsed)
    store = _store(request)
    path = store.output_glb(parsed)
    if not path.exists():
        path = store.model_path(parsed)
    if not record.get("hasModel") or not path.exists():
        raise HTTPException(status_code=404, detail="This scan does not have a reconstructed model yet.")
    return FileResponse(path, media_type="model/gltf-binary", filename="room.glb")


@router.get("/scan/{scan_id}/preview")
def video_preview(scan_id: str, request: Request) -> FileResponse:
    parsed = _parse_id(scan_id)
    _require(request, parsed)
    frames = image_files(_store(request).frames_path(parsed)) if _store(request).frames_path(parsed).exists() else []
    if not frames:
        frames = image_files(_store(request).upload_dir(parsed))
    if not frames:
        raise HTTPException(status_code=404, detail="This scan does not have a preview yet.")
    preview = frames[len(frames) // 2]
    media = "image/png" if preview.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(preview, media_type=media)


def _require(request: Request, scan_id: str) -> dict:
    record = _store(request).get(_parse_id(scan_id))
    if record is None:
        raise HTTPException(status_code=404, detail="This scan could not be found.")
    return record


def _job(record: dict, base_url: str) -> JobStatusResponse:
    scan_id = record["id"]
    has_model = bool(record.get("hasModel"))
    has_preview = bool(record.get("usableFrameCount") or record.get("photoCount") or record.get("source") == "video")
    return JobStatusResponse(
        id=scan_id,
        name=record.get("name") or "New room",
        status=job_status_of(record),
        progress=record.get("progress"),
        processingStage=record.get("processingStage"),
        estimatedSecondsRemaining=record.get("estimatedSecondsRemaining"),
        frameCount=record.get("frameCount"),
        usableFrameCount=record.get("usableFrameCount"),
        durationSeconds=record.get("durationSeconds"),
        modelUrl=f"{base_url}/scan/{scan_id}/model" if has_model else None,
        previewUrl=f"{base_url}/scan/{scan_id}/preview" if has_preview and record.get("usableFrameCount") else None,
        errorMessage=record.get("errorMessage"),
        createdAt=record["createdAt"],
        completedAt=record.get("completedAt"),
    )
