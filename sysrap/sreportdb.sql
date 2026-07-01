
-- Drop child table first to avoid foreign key conflicts
DROP TABLE IF EXISTS opticks_events;
DROP TABLE IF EXISTS opticks_runs;

-- 1. THE PARENT TABLE: Core Run Metadata
CREATE TABLE opticks_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp       INTEGER NOT NULL, -- When the run started

    -- Software & Hardware Environment (Text Metadata)
    software_version    TEXT NOT NULL,
    command_line        TEXT NOT NULL,
    gpu_type            TEXT NOT NULL,
    geometry_name       TEXT NOT NULL,
    geometry_digest     TEXT NOT NULL,

    -- Global Run Metrics / Initialization Timings
    dt_geometry_upload  INTEGER NOT NULL, -- Init timing
    total_events        INTEGER NOT NULL  -- Target total count expected
);
-- STRICT is a SQLite 3.37+ feature

-- 2. THE CHILD TABLE: Granular Per-Event Metrics
CREATE TABLE opticks_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The connection link to the parent run
    run_id           INTEGER NOT NULL,

    -- Your existing array data
    event_index      INTEGER NOT NULL,
    event_timestamp  INTEGER NOT NULL,
    genstep_count    INTEGER NOT NULL,
    photon_count     INTEGER NOT NULL,
    hit_count        INTEGER NOT NULL,
    launch_count     INTEGER NOT NULL,
    dt_upload        INTEGER NOT NULL,
    dt_simulate      INTEGER NOT NULL,
    dt_download      INTEGER NOT NULL,

    -- Enforce the relational integrity
    FOREIGN KEY (run_id) REFERENCES opticks_runs(id) ON DELETE CASCADE
);
-- STRICT is a SQLite 3.37+ feature

-- Performance Indexes
CREATE INDEX idx_events_run_id ON opticks_events(run_id);
CREATE INDEX idx_runs_timestamp ON opticks_runs(run_timestamp);


