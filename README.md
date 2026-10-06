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
- `POST /api/scans/{scanId}/process` → `{ "scanId", "status": "processing" }`
- `GET /api/scans/{scanId}` polling payload used by the app
- `GET /api/scans` history
- `PATCH /api/scans/{scanId}` rename
- `GET /api/scans/{scanId}/model` the GLB
- `GET /api/scans/{scanId}/thumbnail` the first photo

`modelUrl` in scan responses is an absolute URL built from the request host, so a phone on the same Wi-Fi can download the file.

## Phone

Start this server, then start Expo from `../3d`. The app uses the Expo dev-host IP and port 8000. Override that with `EXPO_PUBLIC_API_URL` when the API is somewhere else.

Reconstruction of 10–20 photos often takes several minutes. The processing screen polls real stage updates. It does not animate a fake timer.
# 3dmodel_backend
