#pragma once
/**
sreportdb.h
============



**/

#include "sreport.h"
#include "NSQLite.h"

struct sreportdb
{
    static constexpr const char* SCHEMA = "sreportdb.sql" ;
    std::string db_path ;
    std::string schema_path ;
    std::string schema_sql ;
    NSQLite* db ;

    sreportdb(const char* dbpath);
    std::string desc() const;

    int import_run(const char* runfold);
    int _import_run(   const NP* run, const NP* evsmry);
    int _import_evsmry(const NP* evsmry, int64_t run_id);


};


/**
sreportdb::sreportdb
----------------------

THIS CTOR DROPS PREEXISTING run and evt tables

**/


inline sreportdb::sreportdb(const char* dbpath)
    :
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
    ss << " db_path     " << db_path << "\n" ;
    ss << " schema_path " << schema_path << "\n" ;
    ss << " schema_sql\n" << schema_sql << "\n" ;
    ss << "]sreportdb::desc\n" ;
    std::string str = ss.str() ;
    return str ;
}


inline int sreportdb::import_run(const char* runfold)
{
    const NP* run = NP::Load(runfold, "run.npy");
    if (!run) {
        std::cerr << "sreportdb::import_run Error: Failed to load run.npy array from " << ( runfold ? runfold : "-" ) << "\n";
        return -1;
    }

    const NP* evsmry = NP::Load(runfold, "evsmry.npy");
    if (!evsmry) {
        std::cerr << "sreportdb::import_run Error: Failed to load evsmry.npy array from " << ( runfold ? runfold : "-" ) << "\n";
        return -1;
    }

    return _import_run( run, evsmry );
}

inline int sreportdb::_import_run(const NP* run, const NP* evsmry)
{
    int64_t run_id = 0 ; // TODO: get this from db ingestion
    return _import_evsmry( evsmry, run_id );
}

inline int sreportdb::_import_evsmry(const NP* evsmry, int64_t run_id)
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
        return -2;
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

        inserter.execute(run_id, event_index, event_timestamp, genstep_count,
                         photon_count, hit_count, launch_count, dt_upload,
                         dt_simulate, dt_download);
    }

    // 3. COMMIT THE TRANSACTION
    sqlite3_exec(db->db, "COMMIT;", nullptr, nullptr, nullptr);

    std::cout << "]sreportdb::_import_evsmry\n";
    return 0 ;
}



