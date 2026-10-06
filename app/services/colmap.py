import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np


class ReconstructionError(RuntimeError):
    pass


def reconstruct_points(image_dir: Path, work_dir: Path, on_progress: Callable[[str], None]) -> Path:
    """Run COLMAP sparse reconstruction and return a colored PLY point cloud."""
    image_dir = image_dir.resolve()
    work_dir = work_dir.resolve()
    database = work_dir / "database.db"
    sparse = work_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    ply_path = work_dir / "points.ply"

    binary = _colmap_binary()
    if binary is not None:
        _run_cli(binary, database, image_dir, sparse, ply_path, on_progress)
        return ply_path

    try:
        import pycolmap
    except ImportError as exc:
        raise ReconstructionError(
            "COLMAP is not available. Install the COLMAP application and add it to PATH, or install pycolmap."
        ) from exc

    on_progress("analyzing")
    _run_pycolmap(pycolmap, database, image_dir, sparse, ply_path, on_progress)
    return ply_path


def _colmap_binary() -> str | None:
    candidates = [
        os.environ.get("COLMAP_BIN"),
        shutil.which("colmap"),
        r"C:\Program Files\COLMAP\colmap.exe",
        r"C:\Program Files\COLMAP\COLMAP.bat",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _run_cli(
    binary: str,
    database: Path,
    image_dir: Path,
    sparse: Path,
    ply_path: Path,
    on_progress: Callable[[str], None],
) -> None:
    on_progress("analyzing")
    extract = [
        binary,
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(image_dir),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        "SIMPLE_PINHOLE",
        "--SiftExtraction.max_image_size",
        "2400",
    ]
    params = _camera_params(image_dir)
    if params is not None:
        extract.extend(["--ImageReader.camera_params", params])
    _command(extract)
    on_progress("cameras")
    _command(
        [
            binary,
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.max_num_matches",
            "32768",
        ]
    )
    _command(
        [
            binary,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse),
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.min_focal_length_ratio",
            "0.8",
            "--Mapper.max_focal_length_ratio",
            "1.25",
            "--Mapper.init_min_tri_angle",
            "4",
        ]
    )
    model_dir = _largest_model(sparse)
    try:
        import pycolmap
    except ImportError:
        pycolmap = None
    if pycolmap is not None:
        _reject_degenerate(pycolmap.Reconstruction(model_dir), image_dir)
    on_progress("point-cloud")
    if ply_path.exists():
        ply_path.unlink()
    _command(
        [
            binary,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(ply_path),
            "--output_type",
            "PLY",
        ]
    )
    if not ply_path.exists():
        raise ReconstructionError("COLMAP finished without writing a point cloud.")


def _run_pycolmap(pycolmap, database: Path, image_dir: Path, sparse: Path, ply_path: Path, on_progress: Callable[[str], None]) -> None:
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_PINHOLE"
    params = _camera_params(image_dir)
    if params is not None:
        reader.camera_params = params

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = 2400
    extraction.sift.max_num_features = 12000
    extraction.sift.peak_threshold = 0.004
    pycolmap.extract_features(
        database,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
    )
    _lock_focal_prior(pycolmap, database)

    on_progress("cameras")
    matching = pycolmap.FeatureMatchingOptions()
    sequential = pycolmap.SequentialPairingOptions()
    sequential.overlap = 12
    sequential.quadratic_overlap = True
    pycolmap.match_sequential(database, matching_options=matching, pairing_options=sequential)
    pycolmap.match_exhaustive(database, matching_options=matching)

    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_extra_params = False
    options.ba_refine_focal_length = True
    options.min_focal_length_ratio = 0.8
    options.max_focal_length_ratio = 1.25
    options.min_model_size = 3
    mapper = options.mapper
    mapper.min_focal_length_ratio = 0.8
    mapper.max_focal_length_ratio = 1.25
    mapper.abs_pose_refine_extra_params = False
    mapper.init_min_tri_angle = 4.0
    mapper.init_min_num_inliers = 40
    mapper.abs_pose_min_num_inliers = 20
    mapper.abs_pose_min_inlier_ratio = 0.1
    mapper.max_reg_trials = 12
    mapper.filter_min_tri_angle = 1.0
    options.mapper = mapper

    reconstructions = pycolmap.incremental_mapping(database, image_dir, sparse, options=options)
    if not reconstructions:
        raise ReconstructionError(
            "COLMAP could not register these photos. Step to a new spot between shots and overlap each photo with the last one."
        )
    reconstruction = max(
        reconstructions.values(),
        key=lambda item: (item.num_reg_images(), item.num_points3D()),
    )
    _reject_degenerate(reconstruction, image_dir)
    on_progress("point-cloud")
    if hasattr(reconstruction, "export_PLY"):
        reconstruction.export_PLY(str(ply_path))
        return
    _write_ply(reconstruction, ply_path)


def _lock_focal_prior(pycolmap, database: Path) -> None:
    db = pycolmap.Database.open(database)
    try:
        for camera in db.read_all_cameras():
            camera.has_prior_focal_length = True
            db.update_camera(camera)
    finally:
        db.close()


def _reject_degenerate(reconstruction, image_dir: Path) -> None:
    """Refuse a Poisson sheet when the photos have no real camera motion."""
    image_count = sum(1 for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    registered = int(reconstruction.num_reg_images())
    if registered < 3 or (registered < 8 and registered < image_count * 0.5):
        raise ReconstructionError(
            f"Only {registered} of {image_count} photos could be placed in 3D. "
            "Step sideways to a new spot between photos and keep the previous view in the frame. "
            "Turning in place makes a panorama, not a room."
        )

    baseline, depth = _baseline_and_depth(reconstruction)
    if baseline <= 1e-8 or depth > baseline * 8.0:
        raise ReconstructionError(
            "The camera barely moved between these photos, so the room has no measurable depth. "
            "Walk around the space, overlap each photo with the last, and include the floor and furniture."
        )


def _baseline_and_depth(reconstruction) -> tuple[float, float]:
    centers = []
    posed = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        centers.append(np.asarray(image.projection_center(), dtype=np.float64))
        posed.append(image)
    if len(centers) < 2 or not posed:
        return 0.0, 0.0

    stacked = np.stack(centers)
    baseline = float(np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0)))
    pose = posed[0].cam_from_world()
    rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(pose.translation, dtype=np.float64)
    depths = []
    for point in reconstruction.points3D.values():
        depth = float((rotation @ np.asarray(point.xyz, dtype=np.float64) + translation)[2])
        if depth > 0:
            depths.append(depth)
    if not depths:
        return baseline, 0.0
    depths.sort()
    return baseline, depths[len(depths) // 2]


def _write_ply(reconstruction, ply_path: Path) -> None:
    points = list(reconstruction.points3D.values())
    if len(points) < 30:
        raise ReconstructionError(
            "COLMAP found too few 3D points. Capture more overlapping photos with the same lighting."
        )
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    lines = [header]
    for point in points:
        x, y, z = point.xyz
        red, green, blue = point.color
        lines.append(f"{x} {y} {z} {int(red)} {int(green)} {int(blue)}")
    ply_path.write_text("\n".join(lines), encoding="utf-8")


def _camera_params(image_dir: Path) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    images = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        return None
    with Image.open(images[0]) as image:
        width, height = image.size
    focal = 1.2 * max(width, height)
    return f"{focal},{width / 2},{height / 2}"


def _largest_model(sparse: Path) -> Path:
    models = [path for path in sparse.iterdir() if path.is_dir()]
    if not models:
        raise ReconstructionError(
            "COLMAP could not register these photos. Capture more overlapping views of the same room."
        )

    def rank(path: Path) -> int:
        points = path / "points3D.bin"
        if points.exists():
            return points.stat().st_size
        text = path / "points3D.txt"
        return text.stat().st_size if text.exists() else 0

    return max(models, key=rank)


def _command(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0:
        output = "\n".join((completed.stdout + "\n" + completed.stderr).strip().splitlines()[-25:])
        raise ReconstructionError(f"COLMAP failed while running {args[1]}.\n{output}")
