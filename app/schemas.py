from typing import Literal

from pydantic import BaseModel, Field


class CreateScanRequest(BaseModel):
    name: str = "New room"


class CreateScanResponse(BaseModel):
    scanId: str
    status: Literal["created"] = "created"


class RenameScanRequest(BaseModel):
    name: str = Field(default="", max_length=80)


class PhotoUploadResponse(BaseModel):
    scanId: str
    photoCount: int
    status: Literal["uploaded"] = "uploaded"


class ProcessScanResponse(BaseModel):
    scanId: str
    status: str


class ScanResponse(BaseModel):
    id: str
    name: str
    photos: list[dict[str, str]] = Field(default_factory=list)
    photoCount: int
    status: str
    source: str = "photos"
    jobStatus: str | None = None
    frameCount: int | None = None
    usableFrameCount: int | None = None
    durationSeconds: float | None = None
    warning: str | None = None
    progress: float | None = None
    processingStage: str | None = None
    estimatedSecondsRemaining: int | None = None
    modelUrl: str | None = None
    thumbnailUrl: str | None = None
    previewUrl: str | None = None
    createdAt: str
    completedAt: str | None = None
    errorMessage: str | None = None


class VideoUploadResponse(BaseModel):
    scanId: str
    status: str
    durationSeconds: float
    videoBytes: int
    warning: str | None = None


class JobStatusResponse(BaseModel):
    id: str
    name: str
    status: str
    progress: float | None = None
    processingStage: str | None = None
    estimatedSecondsRemaining: int | None = None
    frameCount: int | None = None
    usableFrameCount: int | None = None
    durationSeconds: float | None = None
    modelUrl: str | None = None
    previewUrl: str | None = None
    errorMessage: str | None = None
    createdAt: str
    completedAt: str | None = None
