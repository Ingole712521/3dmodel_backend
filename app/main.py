from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, scans
from app.services.storage import ScanStore

store = ScanStore()
store.mark_interrupted()

app = FastAPI(title="RoomScan 3D", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.store = store
app.include_router(health.router)
app.include_router(scans.router, prefix="/api")
