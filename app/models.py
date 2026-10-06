from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record(scan_id: str, name: str) -> dict[str, Any]:
    return {
        "id": scan_id,
        "name": name.strip() or "New room",
        "status": "capturing",
        "photoCount": 0,
        "photos": [],
        "progress": None,
        "processingStage": None,
        "estimatedSecondsRemaining": None,
        "createdAt": utc_now(),
        "completedAt": None,
        "errorMessage": None,
        "hasModel": False,
    }
