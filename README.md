# RoomScan 3D reconstruction API

This service turns a set of room photos into a GLB mesh.

```
photos → COLMAP sparse reconstruction → Open3D mesh → GLB
```

There is no stand-in room. If COLMAP cannot register the photos, or Open3D cannot mesh the points, the scan fails with that error.

## Requirements

- Python 3.11 or 3.12 is the safest choice. Open3D and pycolmap may not publish wheels for newer Python versions.
- COLMAP, either:
  - the `colmap` executable on `PATH` (or `COLMAP_BIN` pointing at it), or
  - the `pycolmap` package from `requirements.txt`
- Open3D

COLMAP's dense stereo stage is not required. The mesh is built from the sparse colored point cloud, which still comes from the uploaded photos.

## Run

From this directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

The API listens on `0.0.0.0:8000`.

Check it:

```bash
curl http://127.0.0.1:8000/health
```

## Endpoints

- `GET /health`
- `POST /api/scans` with `{ "name": "New room" }` → `{ "scanId", "status": "created" }`
- `POST /api/scans/{scanId}/photos` multipart field `files` (jpg, jpeg, png)
- `POST /scan/video` multipart field `file` (mp4, mov, m4v). Optional form fields `scanId` and `name`.
- `POST /api/scans/{scanId}/process` → starts photo reconstruction, or requeues a video scan
- `GET /api/scans/{scanId}` polling payload used by the app
- `GET /scan/{scanId}/status` job status (`UPLOADING`, `EXTRACTING_FRAMES`, `RECONSTRUCTING`, `GENERATING_MESH`, `EXPORTING_GLB`, `COMPLETED`, `FAILED`)
- `GET /api/scans` history
- `PATCH /api/scans/{scanId}` rename
- `GET /api/scans/{scanId}/model` and `GET /scan/{scanId}/model` the GLB
- `GET /scan/{scanId}/preview` a representative extracted frame
- `GET /api/scans/{scanId}/thumbnail` the first photo or extracted frame

`modelUrl` in scan responses is an absolute URL built from the request host, so a phone on the same Wi-Fi can download the file.

## Video scans

Upload an MP4, MOV, or M4V walkthrough. The API stores it at `storage/scans/{scanId}/original/`, extracts frames with FFmpeg (default 3 fps), drops blurry and duplicate frames, and keeps about 150–300 images in `storage/scans/{scanId}/images/`.

Frame extraction uses FFmpeg on the host when it is installed, and otherwise runs FFmpeg through Docker Desktop.

Reconstruction uses Docker Desktop by default. The API starts `yeicor/colmap-openmvs:cuda-latest` with `--gpus all` and mounts the scan directory at `/scan`. COLMAP and OpenMVS run inside that container. The existing `reconstruct_points` and `mesh_from_points` functions stay in place for `RECONSTRUCTION_BACKEND=local`. After the container finishes, the API writes `output/room.glb`, `output/room.obj`, and `output/room.ply`.

```
storage/scans/{scanId}/
  original/video.mp4
  images/frame_0001.jpg
  colmap/
  openmvs/
  output/room.glb
```

Scan records stay in `storage/scans/{scanId}.json` and are mirrored to `storage/roomscan.db` (`app/schema.sql`).

Start Docker Desktop, then start this API with `python run.py`. A video scan calls Docker directly. A 60–90 second 1080p walkthrough is the intended input.

`docker compose up --build` is only needed when the API itself should run in a container. Set `SCANS_HOST_PATH` to the host path of `storage/scans` in that case.

## Phone

Start this server, then start Expo from `../3d`. The app uses the Expo dev-host IP and port 8000. Override that with `EXPO_PUBLIC_API_URL` when the API is somewhere else.

Reconstruction of 10–20 photos often takes several minutes. The processing screen polls real stage updates. It does not animate a fake timer.
# 3dmodel_backend
