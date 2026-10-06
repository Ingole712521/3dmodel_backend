import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"

MIN_PHOTOS = 6
MAX_PHOTOS = 30
MAX_IMAGE_BYTES = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}

MAX_VIDEO_BYTES = int(os.environ.get("MAX_VIDEO_BYTES", str(500 * 1024 * 1024)))
MIN_VIDEO_SECONDS = float(os.environ.get("MIN_VIDEO_SECONDS", "8"))
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "180"))
RECOMMENDED_MIN_SECONDS = 60
RECOMMENDED_MAX_SECONDS = 90

FRAME_FPS = float(os.environ.get("FRAME_FPS", "3"))
TARGET_MIN_FRAMES = int(os.environ.get("TARGET_MIN_FRAMES", "150"))
TARGET_MAX_FRAMES = int(os.environ.get("TARGET_MAX_FRAMES", "300"))
BLUR_VARIANCE_MIN = float(os.environ.get("BLUR_VARIANCE_MIN", "60"))
DUPLICATE_SIMILARITY = float(os.environ.get("DUPLICATE_SIMILARITY", "0.985"))

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# docker is the default: Docker Desktop runs yeicor/colmap-openmvs with the GPU.
# Set RECONSTRUCTION_BACKEND=local to use the in-process COLMAP + Open3D functions instead.
RECONSTRUCTION_BACKEND = os.environ.get("RECONSTRUCTION_BACKEND", "docker").strip().lower()
COLMAP_DOCKER_IMAGE = os.environ.get("COLMAP_DOCKER_IMAGE", "yeicor/colmap-openmvs:cuda-latest")
FFMPEG_IMAGE = os.environ.get("FFMPEG_IMAGE", "mwader/static-ffmpeg:latest")
# Host path of storage/scans, required when the API itself runs inside Docker.
SCANS_HOST_PATH = os.environ.get("SCANS_HOST_PATH", "").strip()
DOCKER_TIMEOUT_SECONDS = int(os.environ.get("DOCKER_TIMEOUT_SECONDS", "1500"))
# Keep the finished container so its logs can be inspected with `docker logs`.
DOCKER_DEBUG = os.environ.get("DOCKER_DEBUG", "1").strip().lower() not in {"0", "false", "no"}

JOB_UPLOADING = "UPLOADING"
JOB_EXTRACTING = "EXTRACTING_FRAMES"
JOB_RECONSTRUCTING = "RECONSTRUCTING"
JOB_MESH = "GENERATING_MESH"
JOB_EXPORTING = "EXPORTING_GLB"
JOB_COMPLETED = "COMPLETED"
JOB_FAILED = "FAILED"

ACTIVE_JOBS = {JOB_UPLOADING, JOB_EXTRACTING, JOB_RECONSTRUCTING, JOB_MESH, JOB_EXPORTING}
