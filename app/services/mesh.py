import json
import struct
from pathlib import Path

import numpy as np

from app.services.colmap import ReconstructionError


def mesh_from_points(ply_path: Path, glb_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ReconstructionError(
            "Open3D is not installed. Install the backend requirements with a Python version Open3D supports."
        ) from exc

    cloud = o3d.io.read_point_cloud(str(ply_path))
    if len(cloud.points) < 80:
        raise ReconstructionError(
            "Reconstruction found too few 3D points. Walk the room again and overlap each photo with the last one."
        )

    bounds = cloud.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(bounds.get_extent()))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ReconstructionError("The reconstructed point cloud has no usable size.")

    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    if float(extent.max()) <= 0 or float(extent.min()) / float(extent.max()) < 0.04:
        raise ReconstructionError(
            "These photos reconstruct as a flat sheet, not a room. "
            "Move to a new spot between shots so the walls and floor are seen from more than one place."
        )

    voxel = max(diagonal / 180.0, 1e-4)
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) > 120_000:
        cloud = cloud.voxel_down_sample(voxel * 2.0)
    if len(cloud.points) < 80:
        raise ReconstructionError(
            "The point cloud is too sparse to mesh. Capture more overlapping photos."
        )

    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6.0, max_nn=40))
    try:
        cloud.orient_normals_consistent_tangent_plane(12)
    except RuntimeError:
        cloud.orient_normals_towards_camera_location(cloud.get_center())

    depth = 8 if len(cloud.points) > 2_000 else 7
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    densities = np.asarray(densities)
    if len(densities) == len(mesh.vertices) and len(densities) > 0:
        keep = densities > np.quantile(densities, 0.08)
        mesh.remove_vertices_by_mask(~keep)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) == 0:
        raise ReconstructionError("Open3D could not build a mesh from the reconstructed points.")

    triangle_count = len(mesh.triangles)
    if triangle_count > 80_000:
        mesh = mesh.simplify_quadric_decimation(80_000)
    mesh.compute_vertex_normals()
    _transfer_colors(mesh, cloud)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ReconstructionError("Open3D produced an empty mesh.")

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    write_glb(glb_path, vertices, faces, normals, colors)


def _transfer_colors(mesh, cloud) -> None:
    if not cloud.has_colors():
        return
    import open3d as o3d

    tree = o3d.geometry.KDTreeFlann(cloud)
    source = np.asarray(cloud.colors)
    painted = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    for index, vertex in enumerate(np.asarray(mesh.vertices)):
        _, neighbors, _ = tree.search_knn_vector_3d(vertex, 1)
        painted[index] = source[neighbors[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(painted)


def write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    colors: np.ndarray | None,
) -> None:
    """Serialize a triangle mesh reconstructed by Open3D into a binary glTF file."""
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    normal_array = None if normals is None or len(normals) != len(positions) else np.ascontiguousarray(normals, dtype=np.float32)
    color_array = None
    if colors is not None and len(colors) == len(positions):
        clipped = np.clip(colors, 0.0, 1.0)
        if clipped.shape[1] > 3:
            clipped = clipped[:, :3]
        color_array = np.ascontiguousarray(np.round(clipped * 255.0), dtype=np.uint8)

    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def push(raw: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        views.append(view)
        blob.extend(raw)
        return len(views) - 1

    def add_accessor(buffer_view: int, **fields) -> int:
        accessors.append({"bufferView": buffer_view, **fields})
        return len(accessors) - 1

    position_view = push(positions.tobytes(), 34962)
    position_accessor = add_accessor(
        position_view,
        componentType=5126,
        count=int(len(positions)),
        type="VEC3",
        min=positions.min(axis=0).tolist(),
        max=positions.max(axis=0).tolist(),
    )
    attributes = {"POSITION": position_accessor}
    if normal_array is not None:
        normal_view = push(normal_array.tobytes(), 34962)
        attributes["NORMAL"] = add_accessor(
            normal_view,
            componentType=5126,
            count=int(len(normal_array)),
            type="VEC3",
        )
    if color_array is not None:
        color_view = push(color_array.tobytes(), 34962)
        attributes["COLOR_0"] = add_accessor(
            color_view,
            componentType=5121,
            count=int(len(color_array)),
            type="VEC3",
            normalized=True,
        )

    index_view = push(indices.tobytes(), 34963)
    index_accessor = add_accessor(
        index_view,
        componentType=5125,
        count=int(len(indices)),
        type="SCALAR",
    )

    while len(blob) % 4:
        blob.append(0)

    document = {
        "asset": {"version": "2.0", "generator": "RoomScan3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": index_accessor, "mode": 4}]}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "

    total = 12 + 8 + len(json_chunk) + 8 + len(blob)
    header = struct.pack("<III", 0x46546C67, 2, total)
    json_header = struct.pack("<II", len(json_chunk), 0x4E4F534A)
    bin_header = struct.pack("<II", len(blob), 0x004E4942)
    path.write_bytes(header + json_header + json_chunk + bin_header + bytes(blob))
