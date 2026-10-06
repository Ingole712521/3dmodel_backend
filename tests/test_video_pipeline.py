import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.export import glb_to_obj, obj_to_glb
from app.services.frames import filter_frames
from app.services.mesh import write_glb
from app.utils.files import is_video_container, video_extension


class FrameFilterTests(unittest.TestCase):
    def test_drops_blur_and_duplicates_and_renames(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            _write_noise(folder / "frame_0001.jpg", seed=1)
            _write_noise(folder / "frame_0002.jpg", seed=1)
            _write_flat(folder / "frame_0003.jpg")
            _write_noise(folder / "frame_0004.jpg", seed=2)
            kept = filter_frames(folder, minimum=1)
            names = sorted(path.name for path in folder.glob("frame_*.jpg"))
            self.assertEqual(kept, 2)
            self.assertEqual(names, ["frame_0001.jpg", "frame_0002.jpg"])

    def test_rejects_a_clip_with_too_few_sharp_frames(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            _write_flat(folder / "frame_0001.jpg")
            with self.assertRaises(Exception):
                filter_frames(folder, minimum=2)


class VideoFileTests(unittest.TestCase):
    def test_extension_and_ftyp(self) -> None:
        self.assertEqual(video_extension("room.MP4"), ".mp4")
        self.assertEqual(video_extension("room.mov"), ".mov")
        with self.assertRaises(ValueError):
            video_extension("room.avi")
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "video.mp4"
            path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)
            self.assertTrue(is_video_container(path))
            path.write_bytes(b"not a video file")
            self.assertFalse(is_video_container(path))


class ReportTests(unittest.TestCase):
    def test_mesh_without_glb_is_an_export_failure(self) -> None:
        from app.services.reconstruction_log import (
            apply_file_checkpoints,
            failure_message,
            load_status,
            save_status,
            write_file_manifest,
        )

        with tempfile.TemporaryDirectory() as raw:
            scan = Path(raw)
            mesh = scan / "openmvs"
            mesh.mkdir()
            (mesh / "scene_dense.mvs").write_bytes(b"mvs")
            (mesh / "scene_dense_mesh.ply").write_bytes(b"ply-mesh")
            status = apply_file_checkpoints(scan, {})
            self.assertTrue(status["mesh_generation"])
            self.assertTrue(status["dense_reconstruction"])
            self.assertEqual(failure_message(scan), "Reconstruction succeeded. GLB export failed.")
            save_status(scan, status)
            files = write_file_manifest(scan)
            self.assertIn("openmvs/scene_dense_mesh.ply", files)
            self.assertTrue(load_status(scan)["mesh_generation"])

    def test_missing_mesh_is_a_reconstruction_failure(self) -> None:
        from app.services.reconstruction_log import failure_message

        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(failure_message(Path(raw)), "COLMAP/OpenMVS reconstruction failed.")


class FrameValidationTests(unittest.TestCase):
    def test_skips_a_corrupt_frame_and_records_it(self) -> None:
        from app.services.frames import validate_frames
        from app.services.reconstruction_log import reconstruction_log_path

        with tempfile.TemporaryDirectory() as raw:
            scan = Path(raw)
            images = scan / "images"
            images.mkdir()
            _write_noise(images / "frame_0001.jpg", seed=1)
            (images / "frame_0101.jpg").write_bytes(b"\xff\xd8not-a-jpeg")
            skipped = validate_frames(images, scan)
            self.assertEqual(skipped, ["frame_0101.jpg"])
            self.assertFalse((images / "frame_0101.jpg").exists())
            self.assertTrue((images / "frame_0001.jpg").exists())
            invalid = (scan / "logs" / "invalid_frames.txt").read_text(encoding="utf-8")
            self.assertIn("frame_0101.jpg", invalid)
            self.assertIn("size=", invalid)
            self.assertIn("mime=", invalid)
            valid = (scan / "logs" / "valid_frames.txt").read_text(encoding="utf-8")
            self.assertIn("frame_0001.jpg", valid)
            self.assertIn("width=", valid)
            log = reconstruction_log_path(scan).read_text(encoding="utf-8")
            self.assertIn("INVALID frame_0101.jpg", log)

    def test_aborts_when_every_frame_is_corrupt(self) -> None:
        from app.services.colmap import ReconstructionError
        from app.services.frames import validate_frames

        with tempfile.TemporaryDirectory() as raw:
            scan = Path(raw)
            images = scan / "images"
            images.mkdir()
            (images / "frame_0101.jpg").write_bytes(b"")
            with self.assertRaises(ReconstructionError) as caught:
                validate_frames(images, scan)
            self.assertEqual(str(caught.exception), "Corrupted frame detected: frame_0101.jpg")


class TourTests(unittest.TestCase):
    def test_writes_nodes_links_and_tour_json(self) -> None:
        from app.services.tour import build_tour

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            output.mkdir()
            for index, seed in enumerate((1, 1, 1, 4, 4, 8, 8, 8, 12), start=1):
                _write_noise(images / f"frame_{index:04d}.jpg", seed=seed)

            document = build_tour(images, output, root)

            self.assertEqual(document["kind"], "virtual-tour")
            self.assertEqual(document["startNodeId"], "n0001")
            self.assertGreaterEqual(document["nodeCount"], 3)
            self.assertLessEqual(document["nodeCount"], 9)
            self.assertTrue((output / "tour.json").is_file())
            first = document["nodes"][0]
            self.assertTrue((output / first["image"]).is_file())
            self.assertTrue(any(link["relation"] == "next" for link in first["links"]))
            second = document["nodes"][1]
            self.assertTrue(any(link["direction"] == "back" and link["relation"] == "previous" for link in second["links"]))
            self.assertIn("position", first)
            self.assertEqual(first["position"]["y"], 0.0)


class ExportTests(unittest.TestCase):
    def test_glb_obj_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            glb = folder / "room.glb"
            obj = folder / "room.obj"
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.uint32)
            write_glb(glb, vertices, faces, None, None)
            glb_to_obj(glb, obj)
            obj_to_glb(obj, folder / "again.glb")
            text = obj.read_text(encoding="utf-8")
            self.assertIn("f 1 2 3", text)
            self.assertTrue((folder / "again.glb").stat().st_size > 32)


def _write_noise(path: Path, seed: int) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, size=(180, 180), dtype=np.uint8)
    Image.fromarray(pixels, mode="L").convert("RGB").save(path, quality=95)


def _write_flat(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (180, 180), (140, 140, 140)).save(path, quality=95)


if __name__ == "__main__":
    unittest.main()
