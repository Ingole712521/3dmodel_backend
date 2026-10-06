"""Persistent reconstruction logs under scans/{scan_id}/logs."""

import json
from pathlib import Path

STATUS_KEYS = (
    "feature_extraction",
    "feature_matching",
    "sparse_reconstruction",
    "dense_reconstruction",
    "mesh_generation",
    "mesh_refinement",
    "texture_generation",
    "glb_export",
)


def empty_status() -> dict:
    status = {key: False for key in STATUS_KEYS}
    status["output_files"] = []
    return status


def log_dir(scan_dir: Path) -> Path:
    path = scan_dir / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reconstruction_log_path(scan_dir: Path) -> Path:
    return log_dir(scan_dir) / "reconstruction.log"


def append_log(scan_dir: Path, text: str) -> None:
    if not text:
        return
    path = reconstruction_log_path(scan_dir)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def load_status(scan_dir: Path) -> dict:
    path = log_dir(scan_dir) / "status.json"
    status = empty_status()
    if not path.exists():
        return status
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return status
    if isinstance(loaded, dict):
        for key in STATUS_KEYS:
            status[key] = bool(loaded.get(key))
        files = loaded.get("output_files")
        if isinstance(files, list):
            status["output_files"] = [str(item) for item in files]
    return status


def save_status(scan_dir: Path, status: dict) -> None:
    path = log_dir(scan_dir) / "status.json"
    payload = empty_status()
    payload.update({key: bool(status.get(key)) for key in STATUS_KEYS})
    payload["output_files"] = list(status.get("output_files") or [])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_file_manifest(scan_dir: Path) -> list[str]:
    files = sorted(path.relative_to(scan_dir).as_posix() for path in scan_dir.rglob("*") if path.is_file())
    (log_dir(scan_dir) / "files.txt").write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
    return files


def mesh_ply(scan_dir: Path) -> Path | None:
    preferred = [
        scan_dir / "openmvs" / "scene_dense_mesh_refine.ply",
        scan_dir / "openmvs" / "scene_dense_mesh.ply",
        scan_dir / "output" / "room.ply",
    ]
    for path in preferred:
        if _nonempty(path):
            return path
    found = [
        path
        for path in (scan_dir / "openmvs").glob("*.ply")
        if path.is_file() and path.name != "scene_dense.ply" and path.stat().st_size > 0
    ] if (scan_dir / "openmvs").exists() else []
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


def dense_mvs(scan_dir: Path) -> Path | None:
    preferred = scan_dir / "openmvs" / "scene_dense.mvs"
    if _nonempty(preferred):
        return preferred
    found = _newest(scan_dir / "openmvs", ".mvs")
    return found if found is not None and _nonempty(found) else None


def output_glb(scan_dir: Path) -> Path | None:
    path = scan_dir / "output" / "room.glb"
    if path.exists() and path.stat().st_size >= 32:
        return path
    found = _newest(scan_dir / "output", ".glb") or _newest(scan_dir / "openmvs", ".glb")
    if found is not None and found.stat().st_size >= 32:
        return found
    return None


def apply_file_checkpoints(scan_dir: Path, status: dict) -> dict:
    """A stage counts as successful when its output file exists, even if a later command failed."""
    colmap = scan_dir / "colmap"
    database = colmap / "database.db"
    if _nonempty(database):
        status["feature_extraction"] = True
    sparse = colmap / "sparse"
    if sparse.exists() and any(path.is_dir() and (_nonempty(path / "cameras.bin") or _nonempty(path / "images.bin")) for path in sparse.iterdir()):
        status["sparse_reconstruction"] = True
    if dense_mvs(scan_dir) is not None or _nonempty(scan_dir / "openmvs" / "scene_dense.ply"):
        status["dense_reconstruction"] = True
    if _nonempty(scan_dir / "openmvs" / "scene_dense_mesh.ply"):
        status["mesh_generation"] = True
    if _nonempty(scan_dir / "openmvs" / "scene_dense_mesh_refine.ply"):
        status["mesh_refinement"] = True
    if _newest(scan_dir / "openmvs", ".obj") is not None or _newest(scan_dir / "output", ".obj") is not None:
        status["texture_generation"] = True
    if output_glb(scan_dir) is not None:
        status["glb_export"] = True
    return status


def failure_message(scan_dir: Path) -> str | None:
    mesh = mesh_ply(scan_dir)
    glb = output_glb(scan_dir)
    if mesh is not None and glb is None:
        return "Reconstruction succeeded. GLB export failed."
    if mesh is None:
        return "COLMAP/OpenMVS reconstruction failed."
    return None


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _newest(directory: Path, suffix: str) -> Path | None:
    if not directory.exists():
        return None
    files = [path for path in directory.rglob(f"*{suffix}") if path.is_file() and path.name != f"room{suffix}"]
    if not files:
        room = directory / f"room{suffix}"
        return room if room.is_file() else None
    return max(files, key=lambda path: path.stat().st_mtime)
