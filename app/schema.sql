CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'photos',
    status TEXT NOT NULL,
    job_status TEXT,
    progress REAL,
    processing_stage TEXT,
    photo_count INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER,
    usable_frame_count INTEGER,
    duration_seconds REAL,
    video_filename TEXT,
    video_bytes INTEGER,
    extraction_fps REAL,
    has_model INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status);
CREATE INDEX IF NOT EXISTS idx_scans_job_status ON scans(job_status);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);
