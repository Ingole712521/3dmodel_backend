import shlex
import subprocess
import time
from pathlib import Path

from app.config import COLMAP_DOCKER_IMAGE, DOCKER_DEBUG, DOCKER_TIMEOUT_SECONDS
from app.services.colmap import ReconstructionError
from app.services.docker_cli import docker_bin, docker_mount
from app.services.reconstruction_log import (
    append_log,
    apply_file_checkpoints,
    load_status,
    mesh_ply,
    reconstruction_log_path,
    save_status,
    write_file_manifest,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "reconstruct.sh"


def run_colmap_openmvs(host_scan_dir: Path) -> dict:
    """Run COLMAP and OpenMVS in Docker. A mesh file counts as reconstruction success."""
    host_scan_dir = host_scan_dir.resolve()
    if not host_scan_dir.exists():
        raise ReconstructionError(f"Scan directory {host_scan_dir} does not exist on the Docker host.")

    script = host_scan_dir / "reconstruct.sh"
    script.write_text(
        _SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n"),
        encoding="utf-8",
        newline="\n",
    )
    (host_scan_dir / "logs").mkdir(parents=True, exist_ok=True)
    (host_scan_dir / "output").mkdir(parents=True, exist_ok=True)
    reconstruction_log_path(host_scan_dir).write_text("", encoding="utf-8")
    _replay_frame_reports(host_scan_dir)

    docker = docker_bin()
    container_name = f"roomscan-{host_scan_dir.name}"
    command = [
        docker,
        "run",
        "--gpus",
        "all",
        "--entrypoint",
        "bash",
        "-v",
        f"{docker_mount(host_scan_dir)}:/scan",
    ]
    if DOCKER_DEBUG:
        subprocess.run([docker, "rm", "-f", container_name], capture_output=True, text=True, check=False)
        command.extend(["--name", container_name])
    else:
        command.insert(2, "--rm")
    command.extend([COLMAP_DOCKER_IMAGE, "/scan/reconstruct.sh"])

    started = time.perf_counter()
    append_log(
        host_scan_dir,
        "\n".join(
            [
                "===== docker START =====",
                "COMMAND: " + " ".join(shlex.quote(part) for part in command),
                f"debug_keep_container: {DOCKER_DEBUG}",
            ]
        ),
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=DOCKER_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - started
        _record_docker_result(host_scan_dir, completed.returncode, elapsed, completed.stdout, completed.stderr)
    except FileNotFoundError as exc:
        append_log(host_scan_dir, "Docker Desktop is not running.")
        _finalize_report(host_scan_dir)
        raise ReconstructionError("Docker Desktop is not running. Start Docker Desktop, then try the scan again.") from exc
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        _record_docker_result(
            host_scan_dir,
            None,
            elapsed,
            exc.stdout or "",
            exc.stderr or "",
            timed_out=True,
        )
        if mesh_ply(host_scan_dir) is None:
            raise ReconstructionError("COLMAP and OpenMVS exceeded the reconstruction time limit.") from exc
        append_log(host_scan_dir, "The container timed out after a mesh file was written. Continuing to GLB export.")

    status = _finalize_report(host_scan_dir)
    if mesh_ply(host_scan_dir) is None:
        corruption = _corruption_message(host_scan_dir)
        if corruption:
            raise ReconstructionError(corruption)
        tail = _tail(reconstruction_log_path(host_scan_dir))
        raise ReconstructionError(f"COLMAP/OpenMVS reconstruction failed.\n{tail}")
    return status


def _record_docker_result(
    scan_dir: Path,
    returncode: int | None,
    elapsed: float,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    timed_out: bool = False,
) -> None:
    lines = [
        "===== docker END =====",
        f"exit_code: {returncode if returncode is not None else 'timeout'}",
        f"seconds: {elapsed:.1f}",
        f"timed_out: {timed_out}",
    ]
    out = _text(stdout)
    err = _text(stderr)
    if out.strip():
        lines.extend(["----- docker stdout -----", out.rstrip()])
    if err.strip():
        lines.extend(["----- docker stderr -----", err.rstrip()])
    append_log(scan_dir, "\n".join(lines))


def _finalize_report(scan_dir: Path) -> dict:
    status = apply_file_checkpoints(scan_dir, load_status(scan_dir))
    status["output_files"] = write_file_manifest(scan_dir)
    save_status(scan_dir, status)
    return status


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _replay_frame_reports(scan_dir: Path) -> None:
    logs = scan_dir / "logs"
    chunks = ["===== frame validation ====="]
    for name in ("valid_frames.txt", "invalid_frames.txt"):
        path = logs / name
        text = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
        chunks.append(f"----- {name} -----")
        if text:
            chunks.append(text)
    append_log(scan_dir, "\n".join(chunks))


def _corruption_message(scan_dir: Path) -> str | None:
    path = reconstruction_log_path(scan_dir)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Corrupted frame detected:"):
            return line.strip()
    return None


def _tail(path: Path, limit: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])
