import os
import shutil
import subprocess
from pathlib import Path

from app.config import FFMPEG_BIN, FFMPEG_IMAGE, FFPROBE_BIN, FRAME_FPS
from app.services.colmap import ReconstructionError
from app.services.docker_cli import docker_bin, docker_mount


def resolve_ffmpeg() -> str | None:
    binary = shutil.which(FFMPEG_BIN) or (FFMPEG_BIN if Path(FFMPEG_BIN).exists() else None)
    return binary


def resolve_ffprobe() -> str | None:
    found = shutil.which(FFPROBE_BIN)
    if found:
        return found
    if Path(FFPROBE_BIN).exists():
        return FFPROBE_BIN
    return None


def probe_duration(video_path: Path) -> float:
    ffprobe = resolve_ffprobe()
    if ffprobe is not None:
        completed = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
        )
        if completed.returncode == 0:
            duration = _parse_duration_value(completed.stdout)
            if duration > 0:
                return duration

    ffmpeg = resolve_ffmpeg()
    if ffmpeg is not None:
        completed = _run([ffmpeg, "-i", str(video_path), "-f", "null", "-"])
        duration = _duration_from_ffmpeg(completed.stderr)
        if duration > 0:
            return duration

    completed = _docker_ffmpeg(video_path, video_path.parent, ["-i", _container_path(video_path, video_path.parent), "-f", "null", "-"])
    duration = _duration_from_ffmpeg(completed.stderr + "\n" + completed.stdout)
    if duration <= 0:
        raise ReconstructionError("The video duration could not be read. Upload an MP4, MOV, or M4V file.")
    return duration


def extract_frames(video_path: Path, image_dir: Path, fps: float = FRAME_FPS) -> int:
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rate = fps if fps > 0 else FRAME_FPS
    pattern = image_dir / "frame_%04d.jpg"
    ffmpeg = resolve_ffmpeg()
    if ffmpeg is not None:
        completed = _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-an",
                "-vf",
                f"fps={rate}",
                "-q:v",
                "2",
                str(pattern),
            ]
        )
    else:
        root = _common_root(video_path, image_dir)
        completed = _docker_ffmpeg(
            video_path,
            root,
            [
                "-y",
                "-i",
                _container_path(video_path, root),
                "-an",
                "-vf",
                f"fps={rate}",
                "-q:v",
                "2",
                _container_path(pattern, root),
            ],
        )
    frames = sorted(image_dir.glob("frame_*.jpg"))
    if completed.returncode != 0 or not frames:
        tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-20:])
        raise ReconstructionError(f"FFmpeg could not extract frames from this video.\n{tail}")
    return len(frames)


def _docker_ffmpeg(video_path: Path, root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    root.mkdir(parents=True, exist_ok=True)
    command = [
        docker_bin(),
        "run",
        "--rm",
        "--entrypoint",
        "/ffmpeg",
        "-v",
        f"{docker_mount(root)}:/media",
        "-w",
        "/media",
        FFMPEG_IMAGE,
        *args,
    ]
    try:
        return _run(command)
    except FileNotFoundError as exc:
        raise ReconstructionError("Docker Desktop is not running. Start Docker Desktop, then try the scan again.") from exc


def _common_root(video_path: Path, image_dir: Path) -> Path:
    return Path(os.path.commonpath([video_path.resolve().parent, image_dir.resolve()]))


def _container_path(path: Path, root: Path) -> str:
    return "/media/" + path.resolve().relative_to(root.resolve()).as_posix()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _parse_duration_value(stdout: str) -> float:
    try:
        return float(stdout.strip())
    except ValueError:
        return 0.0


def _duration_from_ffmpeg(stderr: str) -> float:
    for line in stderr.splitlines():
        marker = "Duration:"
        if marker not in line:
            continue
        stamp = line.split(marker, 1)[1].split(",", 1)[0].strip()
        parts = stamp.split(":")
        if len(parts) != 3:
            continue
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
        except ValueError:
            continue
        return hours * 3600 + minutes * 60 + seconds
    return 0.0
