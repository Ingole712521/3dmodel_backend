"""Publish reconstruction files into scans/{id}/output without changing the mesh pipeline."""

import json
import shutil
import struct
from pathlib import Path

import numpy as np

from app.services.colmap import ReconstructionError
from app.services.mesh import write_glb
from app.services.reconstruction_log import mesh_ply, output_glb


def publish_local(ply_path: Path, glb_path: Path, output_dir: Path, model_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    room_glb = output_dir / "room.glb"
    room_ply = output_dir / "room.ply"
    room_obj = output_dir / "room.obj"
    if glb_path.resolve() != room_glb.resolve():
        shutil.copy2(glb_path, room_glb)
    if ply_path.resolve() != room_ply.resolve():
        shutil.copy2(ply_path, room_ply)
    glb_to_obj(room_glb, room_obj)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(room_glb, model_path)


def publish_docker(output_dir: Path, model_path: Path) -> None:
    """Turn whatever OpenMVS managed to write into room.glb, room.obj, and room.ply."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = output_dir.parent
    room_obj = output_dir / "room.obj"
    room_ply = output_dir / "room.ply"
    room_glb = output_dir / "room.glb"
    ply = mesh_ply(scan_dir)
    obj = _newest(scan_dir / "openmvs", ".obj")
    existing = output_glb(scan_dir)
    export_error: Exception | None = None

    if existing is not None and existing.resolve() != room_glb.resolve():
        shutil.copy2(existing, room_glb)
    if obj is not None and not _valid_glb(room_glb):
        try:
            _copy_obj_sidecar(obj, output_dir)
            obj_to_glb(room_obj, room_glb)
        except Exception as exc:
            export_error = exc
    if ply is not None and not _valid_glb(room_glb):
        try:
            mesh_or_cloud_to_glb(ply, room_glb)
        except Exception as exc:
            export_error = exc
    if ply is not None and ply.resolve() != room_ply.resolve():
        shutil.copy2(ply, room_ply)

    if not _valid_glb(room_glb):
        if ply is not None:
            detail = f" {export_error}" if export_error else ""
            raise ReconstructionError(f"Reconstruction succeeded. GLB export failed.{detail}")
        raise ReconstructionError("COLMAP/OpenMVS reconstruction failed.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(room_glb, model_path)


def _valid_glb(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= 32


def mesh_or_cloud_to_glb(source: Path, glb_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ReconstructionError("Open3D is not installed, so the mesh could not be exported to GLB.") from exc

    mesh = o3d.io.read_triangle_mesh(str(source))
    if len(mesh.triangles) > 0 and len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.uint32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None
        write_glb(glb_path, vertices, faces, normals, colors)
        return

    from app.services.mesh import mesh_from_points

    mesh_from_points(source, glb_path)


def obj_to_glb(obj_path: Path, glb_path: Path) -> None:
    vertices, texcoords, faces, texture = _read_obj(obj_path)
    if len(faces) == 0 or len(vertices) == 0:
        raise ReconstructionError("The OpenMVS mesh did not contain any triangles.")
    if texture is None or texcoords is None:
        write_glb(glb_path, vertices, faces, None, None)
        return
    _write_textured_glb(glb_path, vertices, texcoords, faces, texture)


def glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    document, blob = _read_glb(glb_path)
    mesh = document["meshes"][0]["primitives"][0]
    accessors = document["accessors"]
    views = document["bufferViews"]
    positions = _accessor_array(blob, views, accessors, mesh["attributes"]["POSITION"], np.float32, 3)
    indices = _accessor_array(blob, views, accessors, mesh["indices"], np.uint32, 1).reshape(-1, 3)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in positions]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in indices)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_obj_sidecar(obj: Path, output_dir: Path) -> None:
    destination = output_dir / "room.obj"
    text = obj.read_text(encoding="utf-8", errors="replace")
    mtl_names = []
    for line in text.splitlines():
        if line.lower().startswith("mtllib "):
            mtl_names.append(line.split(None, 1)[1].strip())
    rewritten = text
    for name in mtl_names:
        source = obj.parent / name
        if not source.exists():
            continue
        mtl_dest = output_dir / Path(name).name
        mtl_text = source.read_text(encoding="utf-8", errors="replace")
        for token in ("map_Kd", "map_Ka", "map_Bump"):
            mtl_text = _copy_maps(mtl_text, token, source.parent, output_dir)
        mtl_dest.write_text(mtl_text, encoding="utf-8")
        rewritten = rewritten.replace(name, mtl_dest.name)
    destination.write_text(rewritten, encoding="utf-8")


def _copy_maps(mtl_text: str, token: str, source_dir: Path, output_dir: Path) -> str:
    lines = []
    for line in mtl_text.splitlines():
        if line.startswith(token + " "):
            raw = line.split(None, 1)[1].strip()
            source = source_dir / raw
            if source.exists():
                target = output_dir / source.name
                shutil.copy2(source, target)
                line = f"{token} {target.name}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, Path | None]:
    positions: list[list[float]] = []
    uvs: list[list[float]] = []
    corners: list[tuple[int, int | None]] = []
    texture: Path | None = None
    mtl_dir = path.parent
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            parts = line.split()
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("vt "):
            parts = line.split()
            uvs.append([float(parts[1]), float(parts[2])])
        elif line.lower().startswith("mtllib ") and texture is None:
            texture = _texture_from_mtl(mtl_dir / line.split(None, 1)[1].strip())
        elif line.startswith("f "):
            raw = line.split()[1:]
            if len(raw) < 3:
                continue
            face = [_corner(token, len(positions), len(uvs)) for token in raw]
            for index in range(1, len(face) - 1):
                corners.extend((face[0], face[index], face[index + 1]))
    if not positions or not corners:
        return np.zeros((0, 3), np.float32), None, np.zeros((0, 3), np.uint32), texture

    use_uv = all(item[1] is not None for item in corners) and len(uvs) > 0
    packed_v: list[list[float]] = []
    packed_uv: list[list[float]] = []
    triangles: list[list[int]] = []
    lookup: dict[tuple[int, int | None], int] = {}
    triangle: list[int] = []
    for corner in corners:
        if corner[0] < 0 or corner[0] >= len(positions):
            continue
        key = (corner[0], corner[1] if use_uv else None)
        if key not in lookup:
            lookup[key] = len(packed_v)
            packed_v.append(positions[corner[0]])
            if use_uv and corner[1] is not None:
                packed_uv.append(uvs[corner[1]])
        triangle.append(lookup[key])
        if len(triangle) == 3:
            triangles.append(triangle)
            triangle = []
    tex = np.asarray(packed_uv, dtype=np.float32) if use_uv and len(packed_uv) == len(packed_v) else None
    return (
        np.asarray(packed_v, dtype=np.float32),
        tex,
        np.asarray(triangles, dtype=np.uint32),
        texture if tex is not None else None,
    )


def _corner(token: str, vertex_count: int, uv_count: int) -> tuple[int, int | None]:
    parts = token.split("/")
    vertex = _index(parts[0], vertex_count)
    uv = _index(parts[1], uv_count) if len(parts) > 1 and parts[1] else None
    return vertex, uv


def _index(raw: str, count: int) -> int:
    value = int(raw)
    if value < 0:
        return count + value
    return value - 1


def _texture_from_mtl(path: Path) -> Path | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("map_Kd "):
            candidate = path.parent / line.split(None, 1)[1].strip()
            if candidate.exists():
                return candidate
    return None


def _write_textured_glb(
    path: Path,
    vertices: np.ndarray,
    texcoords: np.ndarray,
    faces: np.ndarray,
    texture: Path,
) -> None:
    positions = np.ascontiguousarray(vertices, dtype=np.float32)
    uvs = np.ascontiguousarray(texcoords, dtype=np.float32)
    uvs[:, 1] = 1.0 - uvs[:, 1]
    indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    image = texture.read_bytes()
    mime = "image/png" if image.startswith(b"\x89PNG") else "image/jpeg"

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
    uv_view = push(uvs.tobytes(), 34962)
    uv_accessor = add_accessor(uv_view, componentType=5126, count=int(len(uvs)), type="VEC2")
    index_view = push(indices.tobytes(), 34963)
    index_accessor = add_accessor(index_view, componentType=5125, count=int(len(indices)), type="SCALAR")
    image_view = push(image)

    while len(blob) % 4:
        blob.append(0)

    document = {
        "asset": {"version": "2.0", "generator": "RoomScan3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                }
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor, "TEXCOORD_0": uv_accessor},
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": image_view, "mimeType": mime}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    _write_glb_chunks(path, document, blob)


def _write_glb_chunks(path: Path, document: dict, blob: bytearray) -> None:
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "
    total = 12 + 8 + len(json_chunk) + 8 + len(blob)
    header = struct.pack("<III", 0x46546C67, 2, total)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        header
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(blob), 0x004E4942)
        + bytes(blob)
    )


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ReconstructionError("The exported model is not a GLB file.")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ReconstructionError("The exported model is not a GLB file.")
    document = json.loads(data[20 : 20 + json_len])
    bin_at = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", data, bin_at)
    if bin_type != 0x004E4942:
        raise ReconstructionError("The exported model is missing its geometry.")
    return document, data[bin_at + 8 : bin_at + 8 + bin_len]


def _accessor_array(blob: bytes, views: list[dict], accessors: list[dict], index: int, dtype, width: int) -> np.ndarray:
    accessor = accessors[index]
    view = views[accessor["bufferView"]]
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    component = int(np.dtype(dtype).itemsize)
    raw = blob[start : start + count * width * component]
    array = np.frombuffer(raw, dtype=dtype).reshape(count, width)
    return np.array(array, copy=True)


def _newest(directory: Path, suffix: str) -> Path | None:
    if not directory.exists():
        return None
    files = [path for path in directory.rglob(f"*{suffix}") if path.is_file() and path.name != f"room{suffix}"]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)
