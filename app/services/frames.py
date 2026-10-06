"""Drop blurry, duplicate, and undecodable frames before keyframe selection."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import BLUR_VARIANCE_MIN, DUPLICATE_SIMILARITY, TARGET_MAX_FRAMES, TARGET_MIN_FRAMES
from app.services.colmap import ReconstructionError
from app.services.reconstruction_log import append_log, log_dir


@dataclass
class FrameScore:
    path: Path
    blur: float
    small: np.ndarray


@dataclass
class FrameInspection:
    name: str
    valid: bool
    size: int
    width: int
    height: int
    mime: str
    reason: str = ""

    def report_line(self) -> str:
        line = f"{self.name} size={self.size} width={self.width} height={self.height} mime={self.mime}"
        if self.reason:
            line = f"{line} reason={self.reason}"
        return line


def validate_frames(image_dir: Path, scan_dir: Path) -> list[str]:
    """Remove frames OpenMVS cannot decode. Reports go to logs/valid_frames.txt and logs/invalid_frames.txt."""
    paths = sorted(path for path in image_dir.glob("frame_*.jpg") if path.is_file())
    valid: list[FrameInspection] = []
    invalid: list[FrameInspection] = []
    for path in paths:
        inspection = inspect_frame(path)
        if inspection.valid:
            valid.append(inspection)
            continue
        invalid.append(inspection)
        path.unlink(missing_ok=True)
    _write_frame_reports(scan_dir, valid, invalid)
    _log_frame_reports(scan_dir, valid, invalid)
    if invalid and not valid:
        raise ReconstructionError(f"Corrupted frame detected: {invalid[0].name}")
    return [item.name for item in invalid]


def refresh_valid_frames(image_dir: Path, scan_dir: Path) -> None:
    """Rewrite valid_frames.txt after blur filtering renames the frames COLMAP will read."""
    valid = [inspect_frame(path) for path in sorted(image_dir.glob("frame_*.jpg")) if path.is_file()]
    broken = [item for item in valid if not item.valid]
    for item in broken:
        (image_dir / item.name).unlink(missing_ok=True)
    kept = [item for item in valid if item.valid]
    folder = log_dir(scan_dir)
    (folder / "valid_frames.txt").write_text(_report_text(kept), encoding="utf-8")
    if broken:
        existing = folder / "invalid_frames.txt"
        previous = existing.read_text(encoding="utf-8") if existing.exists() else ""
        existing.write_text(previous + _report_text(broken), encoding="utf-8")
        _log_frame_reports(scan_dir, kept, broken)
        if not kept:
            raise ReconstructionError(f"Corrupted frame detected: {broken[0].name}")


def inspect_frame(path: Path) -> FrameInspection:
    size = path.stat().st_size if path.is_file() else 0
    if size <= 0:
        return FrameInspection(path.name, False, 0, 0, 0, "application/octet-stream", "empty file")
    header = path.read_bytes()[:3]
    if header[:2] != b"\xff\xd8":
        return FrameInspection(path.name, False, size, 0, 0, "application/octet-stream", "invalid JPEG header")
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return FrameInspection(path.name, False, size, 0, 0, "image/jpeg", "failed loading image header")
    if image_format not in {"JPEG", "MPO"}:
        return FrameInspection(path.name, False, size, width, height, "application/octet-stream", "not a JPEG")
    if width <= 0 or height <= 0:
        return FrameInspection(path.name, False, size, width, height, "image/jpeg", "invalid dimensions")
    return FrameInspection(path.name, True, size, width, height, "image/jpeg")


def filter_frames(image_dir: Path, minimum: int) -> int:
    paths = sorted(path for path in image_dir.glob("frame_*.jpg") if path.is_file())
    if not paths:
        raise ReconstructionError("No frames were extracted from the video.")

    scored = [_score(path) for path in paths]
    kept = _select(scored, BLUR_VARIANCE_MIN)
    if len(kept) < TARGET_MIN_FRAMES and len(scored) > len(kept):
        ranked = sorted(item.blur for item in scored)
        relaxed = max(5.0, ranked[max(0, int(len(ranked) * 0.15))])
        kept = _select(scored, min(BLUR_VARIANCE_MIN, relaxed))
    if len(kept) > TARGET_MAX_FRAMES:
        kept = _evenly(kept, TARGET_MAX_FRAMES)
    if len(kept) < minimum:
        raise ReconstructionError(
            f"Only {len(kept)} usable frames remained after removing blur and duplicates. "
            f"Record a slower {TARGET_MIN_FRAMES}-frame walkthrough with steady lighting."
        )
    _rewrite(image_dir, kept)
    return len(kept)


def _score(path: Path) -> FrameScore:
    inspection = inspect_frame(path)
    if not inspection.valid:
        raise ReconstructionError(f"Corrupted frame detected: {path.name}")
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((160, 160))
        array = np.asarray(gray, dtype=np.float32)
    small = np.asarray(Image.fromarray(array.astype(np.uint8)).resize((24, 24)), dtype=np.float32)
    return FrameScore(path=path, blur=_laplacian_variance(array), small=small)


def _write_frame_reports(scan_dir: Path, valid: list[FrameInspection], invalid: list[FrameInspection]) -> None:
    folder = log_dir(scan_dir)
    (folder / "valid_frames.txt").write_text(_report_text(valid), encoding="utf-8")
    (folder / "invalid_frames.txt").write_text(_report_text(invalid), encoding="utf-8")


def _report_text(items: list[FrameInspection]) -> str:
    if not items:
        return ""
    return "\n".join(item.report_line() for item in items) + "\n"


def _log_frame_reports(scan_dir: Path, valid: list[FrameInspection], invalid: list[FrameInspection]) -> None:
    lines = ["===== frame validation ====="]
    lines.extend(f"VALID {item.report_line()}" for item in valid)
    lines.extend(f"INVALID {item.report_line()}" for item in invalid)
    if invalid and valid:
        lines.append(f"Skipped {len(invalid)} corrupted frame(s).")
    if invalid and not valid:
        lines.append(f"Corrupted frame detected: {invalid[0].name}")
    append_log(scan_dir, "\n".join(lines))


def _select(scored: list[FrameScore], blur_min: float) -> list[FrameScore]:
    kept: list[FrameScore] = []
    for item in scored:
        if item.blur < blur_min or float(item.small.mean()) < 8 or float(item.small.mean()) > 248:
            continue
        if kept and _similarity(item.small, kept[-1].small) >= DUPLICATE_SIMILARITY:
            if item.blur > kept[-1].blur:
                kept[-1] = item
            continue
        kept.append(item)
    return kept


def _evenly(items: list[FrameScore], count: int) -> list[FrameScore]:
    if len(items) <= count:
        return items
    indexes = np.linspace(0, len(items) - 1, count)
    chosen: list[FrameScore] = []
    seen: set[int] = set()
    for index in indexes:
        position = int(round(float(index)))
        if position in seen:
            continue
        seen.add(position)
        chosen.append(items[position])
    return chosen


def _rewrite(image_dir: Path, kept: list[FrameScore]) -> None:
    staging = image_dir.parent / "_frames_keep"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    ordered = sorted(kept, key=lambda item: item.path.name)
    for index, item in enumerate(ordered, start=1):
        shutil.copy2(item.path, staging / f"frame_{index:04d}.jpg")
    shutil.rmtree(image_dir)
    staging.rename(image_dir)


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * center
    return float(lap.var())


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = left - float(left.mean())
    b = right - float(right.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 1.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)
