
-- Drop child table first to avoid foreign key conflicts
DROP TABLE IF EXISTS opticks_events;
DROP TABLE IF EXISTS opticks_runs;
DROP TABLE IF EXISTS opticks_versionset;


-- 1. High-Level Version / Hardware Configurations
CREATE TABLE opticks_versionset (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,

    opticks_version    INTEGER NOT NULL,
    geant4_version     INTEGER NOT NULL,
    custom4_version    INTEGER NOT NULL,
    optix_version      INTEGER NOT NULL,

    compute_capability INTEGER NOT NULL,

    cuda_version       INTEGER NOT NULL,
    cuda_driver        INTEGER NOT NULL, -- Resolved at runtime
    nvidia_driver      TEXT NOT NULL,    -- Resolved at runtime

    -- This constraint guarantees uniqueness across the entire set
    UNIQUE(opticks_version, geant4_version, custom4_version, optix_version,
           compute_capability,
           cuda_version, cuda_driver, nvidia_driver)
) ;
-- STRICT is a SQLite 3.37+ feature


-- 1. THE PARENT TABLE: Core Run Metadata
CREATE TABLE opticks_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    versionset_id       INTEGER NOT NULL, -- Foreign Key to the configuration
    run_timestamp       INTEGER NOT NULL, -- When the run started

    fold                TEXT NOT NULL,
    script              TEXT NOT NULL,
    script_arg          TEXT NOT NULL,
    executable          TEXT NOT NULL,

    test                TEXT NOT NULL,
    gpu                 TEXT NOT NULL,
    geometry            TEXT NOT NULL,
    tree_digest         TEXT NOT NULL,

    max_bounce          INTEGER NOT NULL,
    dt_geometry_load    INTEGER NOT NULL,
    dt_geometry_upload  INTEGER NOT NULL,
    total_events        INTEGER NOT NULL,

    ci_pipeline_source  TEXT NOT NULL,           -- blank when not ci
    ci_pipeline_id      INTEGER NOT NULL,        -- -1 when not ci
    ci_job_id           INTEGER NOT NULL,        -- -1 when not ci
    metadata            TEXT NOT NULL,           -- may contain metadata.json

    FOREIGN KEY (versionset_id) REFERENCES opticks_versionset(id)

    -- UNIQUE (ci_pipeline_id, ci_job_id)  -- NULL is never equal to another NULL, but when using -1 this will prevent INSERT

);
-- STRICT is a SQLite 3.37+ feature

DROP INDEX IF EXISTS idx_runs_timestamp;
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON opticks_runs(run_timestamp);

-- 2. Protects local standalone runs from duplicating their timestamps
DROP INDEX IF EXISTS uid_opticks_runs_local_timestamp;
CREATE UNIQUE INDEX IF NOT EXISTS uid_opticks_runs_local_timestamp
ON opticks_runs (run_timestamp)
WHERE ci_pipeline_id = -1;

-- 1. Protects GitLab CI runs from duplicating
DROP INDEX IF EXISTS uid_opticks_runs_ci;
CREATE UNIQUE INDEX IF NOT EXISTS uid_opticks_runs_ci
ON opticks_runs (ci_pipeline_id, ci_job_id)
WHERE ci_pipeline_id != -1;





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










