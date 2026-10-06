from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"

MIN_PHOTOS = 6
MAX_PHOTOS = 30
MAX_IMAGE_BYTES = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
