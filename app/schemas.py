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
    progress: float | None = None
    processingStage: str | None = None
    estimatedSecondsRemaining: int | None = None
    modelUrl: str | None = None
    thumbnailUrl: str | None = None
    createdAt: str
    completedAt: str | None = None
    errorMessage: str | None = None
