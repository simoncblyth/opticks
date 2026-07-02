#pragma once
/**
sreportdb.h
============



**/

#include "ssys.h"
#include "sreport.h"
#include "NSQLite.h"

struct sreportdb
{
    static constexpr const char* _level = "sreportdb__level" ;
    static constexpr const char* SCHEMA = "sreportdb.sql" ;

    int level ;
    std::string db_path ;
    std::string schema_path ;
    std::string schema_sql ;
    NSQLite* db ;

    sreportdb(const char* dbpath);
    std::string desc() const;

    int  import_run(const char* fold);

    int64_t _import_versionset(const NP* run);
    int64_t _import_run(const char* fold, const NP* run, const NP* evsmry, int64_t versionset_id );
    int64_t _import_evsmry(const NP* evsmry, int64_t run_id);

};


/**
sreportdb::sreportdb
----------------------

THIS CTOR EXECUTES SCHEMA SQL THAT DROPS PREEXISTING TABLES

**/



inline sreportdb::sreportdb(const char* dbpath)
    :
    level(ssys::getenvint(_level, 0)),
    db_path(dbpath),
    schema_path(sfilesystem::ExecutablePathSibling(SCHEMA)),
    schema_sql(U::ReadStringDirect(schema_path.c_str())),
    db(new NSQLite(db_path.c_str()))
{
    db->exec(schema_sql.c_str());
}


inline std::string sreportdb::desc() const
{
    std::stringstream ss ;
    ss << "[sreportdb::desc\n" ;
    ss << " level       " << level << "\n" ;
    ss << " db_path     " << db_path << "\n" ;
    ss << " schema_path " << schema_path << "\n" ;
    ss << " schema_sql\n" << schema_sql << "\n" ;
    ss << "]sreportdb::desc\n" ;
    std::string str = ss.str() ;
    return str ;
}


/**
sreportdb::import_run
-----------------------

Most of the run metadata comes from smeta::Collect

**/


inline int sreportdb::import_run(const char* _fold)
{
    std::cout << "[sreportdb::import_run " << ( _fold ? _fold : "-" ) << "\n";
    const NP* run = NP::Load(_fold, "run.npy");
    if (!run) {
        std::cerr << "sreportdb::import_run Error: Failed to load run.npy array from " << ( _fold ? _fold : "-" ) << "\n";
        return -1;
    }

    const NP* evsmry = NP::Load(_fold, "evsmry.npy");
    if (!evsmry) {
        std::cerr << "sreportdb::import_run Error: Failed to load evsmry.npy array from " << ( _fold ? _fold : "-" ) << "\n";
        return -1;
    }

    std::cout << "-sreportdb::import_run run    " << ( run    ? run->sstr() : "-" ) << "\n" ;
    std::cout << "-sreportdb::import_run evsmry " << ( evsmry ? evsmry->sstr() : "-" ) << "\n" ;


    int64_t versionset_id = _import_versionset(run);
    std::cout << "-sreportdb::import_run versionset_id " << versionset_id << "\n" ;
    assert( versionset_id > 0 );

    int64_t run_id        = _import_run(_fold, run, evsmry, versionset_id );

    int rc                = _import_evsmry( evsmry, run_id );

    std::cout << "]sreportdb::import_run " << ( _fold ? _fold : "-" ) << " rc {" << rc << "} " << "\n";
    return rc ;
}



/**


querying with join

SELECT
    r.run_timestamp / 1000000 AS time,
    v.nvidia_driver AS "Driver",
    v.opticks_version AS "Opticks Ver",
    r.dt_geometry_upload AS "Upload Duration"
FROM opticks_runs r
JOIN opticks_versionset v ON r.versionset_id = v.id
ORDER BY r.run_timestamp ASC;

**/


inline int64_t sreportdb::_import_versionset(const NP* run)
{
    std::cout << "[sreportdb::import_versionset run " << ( run ? run->sstr() : "-" ) << "\n";
    int64_t opticks_version    = run->get_meta<int64_t>("OpticksVersion",0);
    int64_t geant4_version     = run->get_meta<int64_t>("Geant4Version",0);
    int64_t optix_version      = run->get_meta<int64_t>("OptiXVersion",0);
    int64_t compute_capability = run->get_meta<int64_t>("ComputeCapability",0);
    int64_t cuda_version       = run->get_meta<int64_t>("CUDAVersion",0);
    int64_t cuda_driver        = run->get_meta<int64_t>("CUDADriver",0);
    std::string nvidia_driver  = run->get_meta<std::string>("NvidiaDriverVersion", "");

   const char* insert_or_ignore_sql =
        "INSERT OR IGNORE INTO opticks_versionset ("
        "opticks_version, geant4_version, optix_version, compute_capability, "
        "cuda_version, cuda_driver, nvidia_driver "
        ") VALUES (?, ?, ?,  ?,  ?, ?, ?);";

    NSQLiteStmt insertor(db->db, insert_or_ignore_sql);
    bool insertor_success = insertor.execute(
                      opticks_version, geant4_version, optix_version, compute_capability,
                      cuda_version, cuda_driver, nvidia_driver
                     );
    assert(insertor_success);


    // 3. Query the ID (Whether newly created or pre-existing)
    const char* select_sql =
        "SELECT id FROM opticks_versionset WHERE"
        " opticks_version = ?"
        " AND"
        " geant4_version = ?"
        " AND"
        " optix_version = ?"
        " AND"
        " compute_capability = ?"
        " AND"
        " cuda_version = ?"
        " AND"
        " cuda_driver = ?"
        " AND"
        " nvidia_driver = ?"
        ";"
        ;

    NSQLiteStmt selector(db->db, select_sql);

    auto result = selector.query_scalar<int64_t>(
                      opticks_version, geant4_version, optix_version, compute_capability,
                      cuda_version, cuda_driver, nvidia_driver
                      );


    int64_t versionset_id = result.has_value() ? result.value() : -1 ;
    if(versionset_id < 0) throw std::runtime_error("sreportdb::_import_versionset FAILED - versionset could not be verified or created.");
    std::cout << "]sreportdb::import_versionset versionset_id " << versionset_id << "\n";
    return versionset_id ;
}


inline int64_t sreportdb::_import_run(const char* _fold, const NP* run, const NP* evsmry, int64_t versionset_id )
{
    std::cout << "[sreportdb::_import_run versionset_id " << versionset_id << "\n";
    int64_t     run_timestamp      = run->get_meta<int64_t>("InitTimestamp",0);   // formerly _init_stamp

    std::string fold               = _fold ;
    std::string script             = run->get_meta<std::string>("SCRIPT","");
    std::string executable         = run->get_meta<std::string>("ExecutableName","");
    std::string test               = run->get_meta<std::string>("TEST","");
    std::string gpu                = run->get_meta<std::string>("GPUMeta","");
    std::string geometry           = run->get_meta<std::string>("GEOM","");
    std::string tree_digest        = run->get_meta<std::string>("TreeDigest","");

    int64_t     max_bounce         = run->get_meta<int64_t>("OPTICKS_MAX_BOUNCE",0);
    int64_t     dt_geometry_load   = evsmry->get_meta<int64_t>("load_geom", 0);
    int64_t     dt_geometry_upload = evsmry->get_meta<int64_t>("upload_geom", 0);
    int64_t     total_events       = evsmry->num_items();

    const char* sql = "INSERT OR REPLACE INTO opticks_runs ("
                      "versionset_id, run_timestamp,"
                      "fold, script, executable, test, gpu, geometry, tree_digest,"
                      "max_bounce, dt_geometry_load, dt_geometry_upload, total_events"
                      ") "
                      "VALUES ("
                      "?,?,"
                      "?, ?, ?, ?, ?, ?, ?,"
                      "?, ?, ?, ?"
                      ");";

    NSQLiteStmt inserter(db->db, sql);
    bool success = inserter.execute(
                      versionset_id, run_timestamp,
                      fold, script, executable, test, gpu, geometry, tree_digest,
                      max_bounce, dt_geometry_load, dt_geometry_upload, total_events);

    assert(success);

    int64_t run_id = sqlite3_last_insert_rowid(db->db);
    std::cout << "]sreportdb::_import_run run_id " << run_id << "\n";
    return run_id ;
}

/**
sreportdb::_import_evsmry
--------------------------

**/

inline int64_t sreportdb::_import_evsmry(const NP* evsmry, int64_t run_id)
{
    std::cout << "[sreportdb::_import_evsmry " << evsmry->sstr() << "\n";
    assert(evsmry->num_dim() == 2);

    const std::vector<std::string>& labels = *(evsmry->labels);
    size_t num_labels = labels.size();

    // 1. Safe Label Extraction Helper
    auto get_label_idx = [&labels, num_labels](const std::string& key) -> size_t {
        size_t idx = std::find(labels.begin(), labels.end(), key) - labels.begin();
        if (idx >= num_labels) {
            throw std::runtime_error("Required column label missing from NP array: " + key);
        }
        return idx;
    };

    size_t index_idx, timestamp_idx, genstep_idx, photon_idx, hit_idx;
    size_t launch_idx, upload_idx, simulate_idx, download_idx;

    try {
        index_idx     = get_label_idx("index");
        timestamp_idx = get_label_idx("timestamp");
        genstep_idx   = get_label_idx("genstep");
        photon_idx    = get_label_idx("photon");
        hit_idx       = get_label_idx("hit");
        launch_idx    = get_label_idx("launch");
        upload_idx    = get_label_idx("dt_upload");
        simulate_idx  = get_label_idx("dt_simulate");
        download_idx  = get_label_idx("dt_download");
    } catch (const std::exception& e) {
        std::cerr << "Schema error: " << e.what() << "\n";
        delete evsmry;
        throw std::runtime_error("sreportdb::_import_evsmry FAILED - missing required label in evsmry array");
    }

    const char* sql = "INSERT OR REPLACE INTO opticks_events (run_id, event_index, event_timestamp, "
                      "genstep_count, photon_count, hit_count, launch_count, dt_upload, dt_simulate, dt_download) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

    NSQLiteStmt inserter(db->db, sql);

    size_t ni = evsmry->shape[0];
    size_t nj = evsmry->shape[1];
    const int64_t* ee = evsmry->cvalues<int64_t>();

    // 2. BEGIN THE TRANSACTION (Massive speedups)
    sqlite3_exec(db->db, "BEGIN TRANSACTION;", nullptr, nullptr, nullptr);

    for (size_t i = 0; i < ni; i++)
    {
        const int64_t* row = ee + (i * nj);

        int64_t event_index     = row[index_idx];
        int64_t event_timestamp = row[timestamp_idx];
        int64_t genstep_count   = row[genstep_idx];
        int64_t photon_count    = row[photon_idx];
        int64_t hit_count       = row[hit_idx];
        int64_t launch_count    = row[launch_idx];
        int64_t dt_upload       = row[upload_idx];
        int64_t dt_simulate     = row[simulate_idx];
        int64_t dt_download     = row[download_idx];

        bool success = inserter.execute(run_id, event_index, event_timestamp, genstep_count,
                                        photon_count, hit_count, launch_count, dt_upload,
                                        dt_simulate, dt_download);
        assert(success);
    }

    // 3. COMMIT THE TRANSACTION
    sqlite3_exec(db->db, "COMMIT;", nullptr, nullptr, nullptr);

    int64_t last_evt_id = sqlite3_last_insert_rowid(db->db);

    std::cout << "]sreportdb::_import_evsmry last_evt_id {" << last_evt_id << "}\n";
    return last_evt_id ;
}


