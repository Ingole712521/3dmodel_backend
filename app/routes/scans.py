from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import MAX_PHOTOS
from app.schemas import (
    CreateScanRequest,
    CreateScanResponse,
    PhotoUploadResponse,
    ProcessScanResponse,
    RenameScanRequest,
    ScanResponse,
)
from app.services.colmap import ReconstructionError
from app.services.reconstruction import image_files, start_reconstruction
from app.services.storage import ScanStore
from app.utils.files import validate_image

router = APIRouter(prefix="/scans")


def _store(request: Request) -> ScanStore:
    return request.app.state.store


def _parse_id(scan_id: str) -> str:
    try:
        return str(UUID(scan_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="This scan could not be found.") from exc


def _public(record: dict, base_url: str) -> ScanResponse:
    scan_id = record["id"]
    model_url = f"{base_url}/api/scans/{scan_id}/model" if record.get("hasModel") else None
    thumbnail_url = None
    photos = record.get("photos") or []
    if photos:
        thumbnail_url = f"{base_url}/api/scans/{scan_id}/thumbnail"
    return ScanResponse(
        id=scan_id,
        name=record["name"],
        photos=[],
        photoCount=int(record.get("photoCount") or 0),
        status=record["status"],
        progress=record.get("progress"),
        processingStage=record.get("processingStage"),
        estimatedSecondsRemaining=record.get("estimatedSecondsRemaining"),
        modelUrl=model_url,
        thumbnailUrl=thumbnail_url,
        createdAt=record["createdAt"],
        completedAt=record.get("completedAt"),
        errorMessage=record.get("errorMessage"),
    )


def _base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@router.post("", response_model=CreateScanResponse)
def create_scan(body: CreateScanRequest, request: Request) -> CreateScanResponse:
    scan_id = str(uuid4())
    _store(request).create(scan_id, body.name)
    return CreateScanResponse(scanId=scan_id, status="created")


@router.get("", response_model=list[ScanResponse])
def list_scans(request: Request) -> list[ScanResponse]:
    base = _base(request)
    return [_public(record, base) for record in _store(request).list_scans()]


@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: str, request: Request) -> ScanResponse:
    record = _require(request, scan_id)
    return _public(record, _base(request))


@router.patch("/{scan_id}", response_model=ScanResponse)
def rename_scan(scan_id: str, body: RenameScanRequest, request: Request) -> ScanResponse:
    _require(request, scan_id)
    name = body.name.strip() or "Untitled room"
    record = _store(request).update(_parse_id(scan_id), name=name)
    return _public(record, _base(request))


@router.post("/{scan_id}/photos", response_model=PhotoUploadResponse)
async def upload_photos(
    scan_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    replace: bool = Query(False),
) -> PhotoUploadResponse:
    parsed = _parse_id(scan_id)
    record = _require(request, parsed)
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one photo to upload.")

    store = _store(request)
    if replace:
        folder = store.clear_uploads(parsed)
        saved: list[dict[str, str]] = []
    else:
        folder = store.upload_dir(parsed)
        saved = list(record.get("photos") or [])
    if len(saved) + len(files) > MAX_PHOTOS:
        raise HTTPException(status_code=400, detail=f"A scan can include at most {MAX_PHOTOS} photos.")

    for upload in files:
        payload = await upload.read()
        original = upload.filename or f"photo-{len(saved) + 1}.jpg"
        try:
            extension = validate_image(original, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stored_name = f"{len(saved) + 1:02d}{extension}"
        (folder / stored_name).write_bytes(payload)
        saved.append({"filename": stored_name})

    store.update(
        parsed,
        photos=saved,
        photoCount=len(saved),
        status="reviewing",
        progress=10,
        processingStage="uploading",
        errorMessage=None,
        hasModel=False,
    )
    return PhotoUploadResponse(scanId=parsed, photoCount=len(saved), status="uploaded")


@router.post("/{scan_id}/process", response_model=ProcessScanResponse)
def process_scan(scan_id: str, request: Request) -> ProcessScanResponse:
    parsed = _parse_id(scan_id)
    record = _require(request, parsed)
    images = image_files(_store(request).upload_dir(parsed))
    if len(images) != int(record.get("photoCount") or 0):
        _store(request).update(parsed, photoCount=len(images))
    try:
        started = start_reconstruction(_store(request), parsed)
    except ReconstructionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="This scan could not be found.") from exc
    return ProcessScanResponse(scanId=parsed, status=started["status"])


@router.get("/{scan_id}/model")
def download_model(scan_id: str, request: Request) -> FileResponse:
    parsed = _parse_id(scan_id)
    record = _require(request, parsed)
    path = _store(request).model_path(parsed)
    if not record.get("hasModel") or not path.exists():
        raise HTTPException(status_code=404, detail="This scan does not have a reconstructed model yet.")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{parsed}.glb")


@router.get("/{scan_id}/thumbnail")
def download_thumbnail(scan_id: str, request: Request) -> FileResponse:
    parsed = _parse_id(scan_id)
    _require(request, parsed)
    images = image_files(_store(request).upload_dir(parsed))
    if not images:
        raise HTTPException(status_code=404, detail="This scan does not have a thumbnail.")
    media = "image/png" if images[0].suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(images[0], media_type=media)


def _require(request: Request, scan_id: str) -> dict:
    record = _store(request).get(_parse_id(scan_id))
    if record is None:
        raise HTTPException(status_code=404, detail="This scan could not be found.")
    return record
