import shutil
import subprocess
from pathlib import Path

from app.services.colmap import ReconstructionError


def docker_bin() -> str:
    found = shutil.which("docker")
    if not found:
        raise ReconstructionError("Docker Desktop is not running. Start Docker Desktop, then try the scan again.")
    return found


def docker_mount(path: Path) -> str:
    """Windows path Docker Desktop can bind into a Linux container."""
    return path.resolve().as_posix()
