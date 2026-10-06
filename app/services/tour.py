"""Virtual tour: keyframes, navigation nodes, a movement graph, and tour.json.

Video frames are already extracted. This stage does not run COLMAP or OpenMVS.
Nodes are viewpoints along the walk. Links are the arrows a Street View or
Matterport tour uses to move between them.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app.config import KEYFRAME_MAX_GAP, KEYFRAME_SIMILARITY, LOOP_SIMILARITY, TOUR_MAX_NODES, TOUR_MIN_NODES
from app.services.colmap import ReconstructionError
from app.services.frames import FrameScore, _score, _similarity
from app.services.reconstruction_log import append_log

NODE_EDGE = 1600


def build_tour(
    image_dir: Path,
    output_dir: Path,
    scan_dir: Path,
    on_stage: Callable[[str], None] | None = None,
) -> dict:
    """Write output/tour.json and output/nodes from the frames in image_dir."""
    frames = sorted(path for path in image_dir.glob("frame_*.jpg") if path.is_file())
    if not frames:
        raise ReconstructionError("No frames were available for the tour.")

    _stage(scan_dir, on_stage, "keyframes")
    chosen = select_keyframes([_score(path) for path in frames])

    nodes_dir = output_dir / "nodes"
    if nodes_dir.exists():
        shutil.rmtree(nodes_dir)
    nodes_dir.mkdir(parents=True, exist_ok=True)

    _stage(scan_dir, on_stage, "nodes")
    nodes, thumbs, headings = _materialize(chosen, nodes_dir)

    _stage(scan_dir, on_stage, "graph")
    _link_path(nodes, thumbs, headings)
    _link_loops(nodes, thumbs, headings)

    _stage(scan_dir, on_stage, "tour")
    document = {
        "version": 1,
        "kind": "virtual-tour",
        "startNodeId": nodes[0]["id"],
        "nodeCount": len(nodes),
        "nodes": nodes,
    }
    destination = output_dir / "tour.json"
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    append_log(scan_dir, f"tour.json nodes={len(nodes)} path={destination.name}")
    return document


def select_keyframes(scored: list[FrameScore]) -> list[FrameScore]:
    """Keep a viewpoint when the picture changes, and at least every few frames."""
    if not scored:
        raise ReconstructionError("No frames were available for the tour.")

    chosen = [scored[0]]
    for offset, item in enumerate(scored[1:], start=1):
        last_offset = _index_of(scored, chosen[-1])
        gap = offset - last_offset
        similarity = _similarity(item.small, chosen[-1].small)
        if similarity >= KEYFRAME_SIMILARITY and gap < KEYFRAME_MAX_GAP:
            if item.blur > chosen[-1].blur and similarity >= 0.95:
                chosen[-1] = item
            continue
        chosen.append(item)

    if len(chosen) > TOUR_MAX_NODES:
        chosen = _evenly(chosen, TOUR_MAX_NODES)
    if len(chosen) < min(TOUR_MIN_NODES, len(scored)):
        chosen = _evenly(scored, min(len(scored), TOUR_MIN_NODES))
    return chosen


def _materialize(chosen: list[FrameScore], nodes_dir: Path) -> tuple[list[dict], list[np.ndarray], list[float]]:
    nodes: list[dict] = []
    thumbs: list[np.ndarray] = []
    headings: list[float] = []
    heading = 0.0
    x = 0.0
    z = 0.0

    for index, item in enumerate(chosen):
        if index > 0:
            pan = _horizontal_shift(chosen[index - 1].small, item.small)
            similarity = _similarity(chosen[index - 1].small, item.small)
            heading = (heading + pan * 35.0) % 360.0
            step = 0.35 + (1.0 - max(0.0, min(1.0, similarity))) * 1.4
            radians = math.radians(heading)
            x += math.sin(radians) * step
            z += math.cos(radians) * step

        node_id = f"n{index + 1:04d}"
        width, height = _write_node(item.path, nodes_dir / f"{node_id}.jpg")
        nodes.append(
            {
                "id": node_id,
                "index": index,
                "frame": item.path.name,
                "image": f"nodes/{node_id}.jpg",
                "width": width,
                "height": height,
                "heading": round(heading, 2),
                "position": {"x": round(x, 3), "y": 0.0, "z": round(z, 3)},
                "links": [],
            }
        )
        thumbs.append(item.small)
        headings.append(heading)
    return nodes, thumbs, headings


def _link_path(nodes: list[dict], thumbs: list[np.ndarray], headings: list[float]) -> None:
    for index in range(len(nodes) - 1):
        pan = _horizontal_shift(thumbs[index], thumbs[index + 1])
        yaw = _wrap(headings[index + 1] - headings[index])
        _add_link(nodes[index], nodes[index + 1]["id"], _step_direction(pan), "next", yaw)
        _add_link(nodes[index + 1], nodes[index]["id"], "back", "previous", _wrap(yaw + 180.0))


def _link_loops(nodes: list[dict], thumbs: list[np.ndarray], headings: list[float]) -> None:
    """Connect viewpoints that look alike even when the walk passed between them."""
    candidates: list[tuple[float, int, int]] = []
    for left in range(len(nodes)):
        for right in range(left + 3, len(nodes)):
            similarity = _similarity(thumbs[left], thumbs[right])
            if similarity >= LOOP_SIMILARITY:
                candidates.append((similarity, left, right))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used: set[int] = set()
    for _similarity_value, left, right in candidates:
        if left in used or right in used:
            continue
        yaw = _wrap(headings[right] - headings[left])
        _add_link(nodes[left], nodes[right]["id"], "nearby", "loop", yaw)
        _add_link(nodes[right], nodes[left]["id"], "nearby", "loop", _wrap(yaw + 180.0))
        used.add(left)
        used.add(right)


def _add_link(node: dict, target: str, direction: str, relation: str, yaw: float) -> None:
    if any(link.get("target") == target for link in node["links"]):
        return
    node["links"].append(
        {
            "target": target,
            "direction": direction,
            "relation": relation,
            "yaw": round(yaw, 2),
        }
    )


def _step_direction(pan: float) -> str:
    if pan > 0.28:
        return "right"
    if pan < -0.28:
        return "left"
    return "forward"


def _horizontal_shift(previous: np.ndarray, current: np.ndarray) -> float:
    """Normalized pan from -1 (left) to 1 (right), estimated by sliding the thumbnails."""
    best_offset = 0
    best_score = -2.0
    width = int(previous.shape[1])
    limit = max(1, width // 4)
    for offset in range(-limit, limit + 1):
        if offset < 0:
            left = previous[:, :offset]
            right = current[:, -offset:]
        elif offset > 0:
            left = previous[:, offset:]
            right = current[:, :-offset]
        else:
            left = previous
            right = current
        score = _similarity(left, right)
        if score > best_score:
            best_score = score
            best_offset = offset
    return best_offset / float(limit)


def _write_node(source: Path, destination: Path) -> tuple[int, int]:
    with Image.open(source) as image:
        frame = image.convert("RGB")
        frame.thumbnail((NODE_EDGE, NODE_EDGE))
        frame.save(destination, format="JPEG", quality=85)
        return frame.size


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


def _index_of(items: list[FrameScore], target: FrameScore) -> int:
    for index, item in enumerate(items):
        if item is target:
            return index
    return 0


def _wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _stage(scan_dir: Path, on_stage: Callable[[str], None] | None, name: str) -> None:
    append_log(scan_dir, f"===== {name} =====")
    if on_stage is not None:
        on_stage(name)
