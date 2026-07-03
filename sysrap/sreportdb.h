#pragma once
/**
sreportdb.h
============



**/


#include <iostream>
#include <filesystem>
#include <string>
#include <regex>

#include "ssys.h"
#include "sreport.h"
#include "NSQLite.h"

struct sreportdb
{
    static bool IsExistingDir(const char* _dir);

    template<typename ... Args> static bool IsExistingDirWith_0(const char* _dir, const Args&... names_ );
    template<typename ... Args> static bool IsExistingDirWith_1(const char* _dir, const Args&... names_ );
    template<typename ... Args> static bool IsExistingDirWith(  const char* _dir, const Args&... names_ );

    static bool IsReportDir( const char* _dir);

    static constexpr const char* _level = "sreportdb__level" ;
    static constexpr const char* SCHEMA = "sreportdb.sql" ;

    int level ;
    std::string db_path ;
    std::string schema_path ;
    std::string schema_sql ;
    NSQLite* db ;

    sreportdb(const char* dbpath);
    std::string desc() const;

    int  import_auto(   const char* dir);
    int  import_archive(const char* archive_dir);
    int  import_report( const char* report_dir);

    int64_t _import_versionset(const NP* run);
    int64_t _import_run(const char* fold, const NP* run, const NP* evsmry, int64_t versionset_id );
    int64_t _import_evsmry(const NP* evsmry, int64_t run_id);

};



bool sreportdb::IsExistingDir(const char* _dir) // static
{
    namespace fs = std::filesystem;
    fs::path dir(_dir);
    return fs::exists(dir) && fs::is_directory(dir);
}


template<typename ... Args>
bool sreportdb::IsExistingDirWith_0(const char* _dir,  const Args&... names_ ) // static
{
    namespace fs = std::filesystem;
    fs::path dir(_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;

    std::vector<std::string_view> names = { names_... };

    for (const auto& name : names)
    {
        fs::path name_file = dir / name;
        if (!fs::exists(name_file) || !fs::is_regular_file(name_file)) {
            return false;
        }
    }
    return true;
}

template<typename ... Args>
bool sreportdb::IsExistingDirWith_1(const char* _dir, const Args&... names)
{
    namespace fs = std::filesystem;
    fs::path dir(_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;

    // C++17 Fold Expression: Short-circuiting verification loop
    return ( (fs::exists(dir / names) && fs::is_regular_file(dir / names)) && ... );
}


template<typename ... Args>
bool sreportdb::IsExistingDirWith(const char* _dir, const Args&... names)
{
    namespace fs = std::filesystem;
    fs::path dir(_dir);
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;

    auto check_file = [&dir](const auto& name) {
        fs::path target_path = dir / name;
        return fs::exists(target_path) && fs::is_regular_file(target_path);
    };

    // C++17 fold expression
    return (check_file(names) && ...);
}


bool sreportdb::IsReportDir(const char* _dir) // static
{
    return IsExistingDirWith(_dir, "run.npy", "evsmry.npy" );
}







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
sreportdb::import_auto
------------------------

If the argument directory is a report dir (ie it contains "run.npy" and "evsmry.npy")
then invoke import_report on it, otherwise invoke import_archive which does a recursive
traverse of the directory tree, looking for directories with names like "sreport_000"
that contain the expected report files. When found invoke import_report.

HMM: note that the directory name check is not done when the argument dir is a report dir...

**/

inline int sreportdb::import_auto(const char* _dir)
{
    return IsReportDir(_dir) ? import_report(_dir) : import_archive(_dir);
}


inline int sreportdb::import_archive(const char* _archive_dir)
{
    std::cout << "[sreportdb::import_archive _archive_dir " << ( _archive_dir ? _archive_dir : "-" ) << "\n";

    if(!IsExistingDir(_archive_dir))
    {
        std::cerr << "-sreportdb::import_archive ERROR : Provided path is not an existing directory.\n";
        return -1 ;
    }

    namespace fs = std::filesystem;
    fs::path archive_dir(_archive_dir);

    std::regex sreport_pattern(R"(^sreport_\d+$)"); // HMM: could skip this name check
    auto iterator_options = fs::directory_options::skip_permission_denied;
    int rc = 0 ;
    for (const auto& entry : fs::recursive_directory_iterator(archive_dir, iterator_options))
    {
        if (!fs::is_directory(entry.path())) continue ;
        if (!std::regex_match(entry.path().filename().string(), sreport_pattern)) continue ;

        std::string entry_dir = entry.path().string();
        const char* _entry_dir = entry_dir.c_str();

        bool is_report_dir = IsReportDir(_entry_dir);
        if(!is_report_dir) continue ;

        int irc = import_report( _entry_dir );
        rc += irc ;

        if( irc != 0 )
        {
            std::cout << "-sreportdb::import_archive FAIL for dir [" << ( _entry_dir ? _entry_dir : "-" ) << "]\n";
            return rc ;
        }
    }
    return rc ;
}



/**
sreportdb::import_report
--------------------------

Most of the run metadata comes from smeta::Collect

**/


inline int sreportdb::import_report(const char* _fold)
{
    std::cout << "[sreportdb::import_report " << ( _fold ? _fold : "-" ) << "\n";
    const NP* run = NP::Load(_fold, "run.npy");
    if (!run) {
        std::cerr << "sreportdb::import_report Error: Failed to load run.npy array from " << ( _fold ? _fold : "-" ) << "\n";
        return -1;
    }

    const NP* evsmry = NP::Load(_fold, "evsmry.npy");
    if (!evsmry) {
        std::cerr << "sreportdb::import_report Error: Failed to load evsmry.npy array from " << ( _fold ? _fold : "-" ) << "\n";
        return -1;
    }

    std::cout << "-sreportdb::import_report run    " << ( run    ? run->sstr() : "-" ) << "\n" ;
    std::cout << "-sreportdb::import_report evsmry " << ( evsmry ? evsmry->sstr() : "-" ) << "\n" ;


    int64_t versionset_id = _import_versionset(run);
    std::cout << "-sreportdb::import_report versionset_id " << versionset_id << "\n" ;
    assert( versionset_id > 0 );

    int64_t run_id        = _import_run(_fold, run, evsmry, versionset_id );

    int64_t last_evt_id   = _import_evsmry( evsmry, run_id );

    std::cout << "]sreportdb::import_report " << ( _fold ? _fold : "-" ) << " last_evt_id {" << last_evt_id << "} " << "\n";
    return 0 ;
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
    std::string script             = run->get_meta<std::string>("TESTSCRIPT|SCRIPT","");
    std::string script_arg         = run->get_meta<std::string>("TESTSCRIPT_ARG|SCRIPT_ARG","");
    std::string executable         = run->get_meta<std::string>("ExecutableName","");
    std::string test               = run->get_meta<std::string>("TEST","");
    std::string gpu                = run->get_meta<std::string>("GPUMeta","");
    std::string geometry           = run->get_meta<std::string>("GEOM","");
    std::string tree_digest        = run->get_meta<std::string>("TreeDigest","");

    int64_t     max_bounce         = run->get_meta<int64_t>("OPTICKS_MAX_BOUNCE",0);
    int64_t     dt_geometry_load   = evsmry->get_meta<int64_t>("load_geom", 0);
    int64_t     dt_geometry_upload = evsmry->get_meta<int64_t>("upload_geom", 0);
    int64_t     total_events       = evsmry->num_items();

    std::string ci_pipeline_source = run->get_meta<std::string>("CI_PIPELINE_SOURCE","");
    int64_t ci_pipeline_id         = run->get_meta<int64_t>("CI_PIPELINE_ID",-1);
    int64_t ci_job_id              = run->get_meta<int64_t>("CI_JOB_ID",-1);

    // HMM: MAYBE A JSON BLOB TOO - FOR CHANGE ISOLATION

    const char* sql = "INSERT OR REPLACE INTO opticks_runs ("
                      "versionset_id, run_timestamp,"
                      "fold, script, script_arg, executable,"
                      "test, gpu, geometry, tree_digest,"
                      "max_bounce, dt_geometry_load, dt_geometry_upload, total_events,"
                      "ci_pipeline_source, ci_pipeline_id, ci_job_id"
                      ") "
                      "VALUES ("
                      "?,?,"
                      "?, ?, ?, ?, "
                      "?, ?, ?, ?, "
                      "?, ?, ?, ?, "
                      "?, ?, ?"
                      ");";

    NSQLiteStmt inserter(db->db, sql);
    bool success = inserter.execute(
                      versionset_id, run_timestamp,
                      fold, script, script_arg, executable,
                      test, gpu, geometry, tree_digest,
                      max_bounce, dt_geometry_load, dt_geometry_upload, total_events,
                      ci_pipeline_source, ci_pipeline_id, ci_job_id);

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


