#pragma once
/**
sreportdb.h - extracts info from run and evsmry arrays and metadata to populate sqlite3 database
================================================================================================

This is included by: sysrap/tests/sreportdb.cc

For tips on development of Grafana queries against the sreportdb database see:

* ~/o/sysrap/sreportdb_grafana.rst


**/


#include <iostream>
#include <filesystem>
#include <string>
#include <regex>
#include <vector>
#include <algorithm>



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
    static std::string LoadMetadata(const char* _fold, const char* name );

    static constexpr const char* _level = "sreportdb__level" ;
    static constexpr const char* SCHEMA = "sreportdb.sql" ;
    static constexpr const char* METADATA = "metadata.json" ;

    int level ;
    const char* dbfold ;
    std::string schema_path ;
    std::string schema_sql ;
    NSQLite* db ;

    static constexpr const char* DBNAME="sreportdb.sqlite3" ;
    sreportdb(const char* dbfold);
    std::string desc() const;

    int  import_auto(   const char* dir);
    int  import_archive(const char* archive_dir);
    int  import_report( const char* report_dir);

    bool is_imported( const NP* run );

    bool is_imported_combi( int64_t ci_pipeline_id, int64_t ci_job_id, int64_t run_timestamp );
    bool is_imported_nonci( int64_t run_timestamp );
    bool is_imported_ci( int64_t ci_pipeline_id, int64_t ci_job_id );



    int64_t _import_versionset(const NP* run);
    int64_t _import_run(const char* fold, const NP* run, const NP* evsmry, int64_t versionset_id, const std::string& metadata );
    std::string desc_runs() const ;

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


inline std::string sreportdb::LoadMetadata(const char* _fold, const char* name ) // static
{
    namespace fs = std::filesystem;
    fs::path dir(_fold);
    fs::path path = dir / name ;
    std::string metadata = "{}" ;
    if (fs::exists(path) && fs::is_regular_file(path))
    {
        std::string _path = path.string();
        std::string content = U::ReadStringDirect( _path.c_str()) ;
        bool looks_like_json = !content.empty() && content.front() == '{' && ( content.back() == '}' || content.back() == '\n' ) ;
        if(looks_like_json) metadata = content;
        if(!looks_like_json) std::cerr
            << "sreportdb::LoadMetadata"
            << " WARNING - malformed json read from [" << _path << "]"
            << " content(" << content << ")"
            << " looks_like_json:" << ( looks_like_json ? "YES" : "NO " )
            << "\n"
            ;
    }
    return metadata ;
}




/**
sreportdb::sreportdb
----------------------

Constructor:

1. loads sql string from .sql file that is sibling to the executable path
2. instanciates NSQLite with provided dbpath
3. executes the schema sql creating tables

   * CURRENTLY THE SCHEMA SQL DROPS PREEXISTING TABLES
   * THIS IS APPROPRIATE WHILE THE SCHEMA IS IN DEVELOPMENT
   * DB REGARDED AS TRANSIENT CACHE OF THE persisted report folders

**/


inline sreportdb::sreportdb(const char* dbfold_)
    :
    level(ssys::getenvint(_level, 0)),
    dbfold(dbfold_ ? strdup(dbfold_) : nullptr),
    schema_path(sfilesystem::ExecutablePathSibling(SCHEMA)),
    schema_sql(U::ReadStringDirect(schema_path.c_str())),
    db(new NSQLite(dbfold,DBNAME))
{
    db->exec(schema_sql.c_str());
}


inline std::string sreportdb::desc() const
{
    std::stringstream ss ;
    ss << "[sreportdb::desc\n" ;
    ss << " level       " << level << "\n" ;
    ss << " dbfold      " << ( dbfold ? dbfold : "-" ) << "\n" ;
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


/**
sreportdb::import_archive
-------------------------

Recursively iterate over subdirectories of  _archive_dir looking
for directories with name and contents of a report directory.
When found invoke sreportdb::import_report populating the sqlite3 database.

**/


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

    //const char* pattern = "" ; // empty pattern yields regex that matches any folder name
    const char* pattern = R"(^sreport_\d+$)" ;  // R means raw string - so no escaping of backslashes
    std::regex sreport_pattern(pattern);

    std::vector<fs::path> paths;
    auto iterator_options = fs::directory_options::skip_permission_denied;
    int rc = 0 ;
    for (const auto& entry : fs::recursive_directory_iterator(archive_dir, iterator_options))
    {
        if (!fs::is_directory(entry.path())) continue ;
        if (!std::regex_match(entry.path().filename().string(), sreport_pattern)) continue ;

        std::string entry_dir = entry.path().string();
        const char* _entry_dir = entry_dir.c_str();
        bool is_report_dir = IsReportDir(_entry_dir);
        if(is_report_dir) paths.push_back(entry.path());
    }

    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths)
    {
        std::cout  << "-sreportdb::import_archive [" << path.string().c_str() << "]\n" ;
    }

    for (const auto& path : paths)
    {
        std::string report_dir = path.string();
        const char* _report_dir = report_dir.c_str();
        //const char* _report_dir = path.string().c_str(); LEADS TO FUNNY CHARS IN DIR - AS GETS POINTER FROM A TEMPORARY
        int irc = import_report( _report_dir );
        rc += irc ;

        if( irc != 0 )
        {
            std::cout << "-sreportdb::import_archive FAIL for dir [" << ( _report_dir ? _report_dir : "-" ) << "]\n";
            return rc ;
        }
    }

    return rc ;
}


/**
sreportdb::import_report
--------------------------

1. loads run and evsmry arrays from fold directory
2. invokes the various _import methods converting array data and metadata into sqlite3 tables

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

    std::string metadata = LoadMetadata(_fold, METADATA );


    if(level > 1) std::cout << "-sreportdb::import_report run    " << ( run    ? run->sstr() : "-" ) << "\n" ;
    if(level > 1) std::cout << "-sreportdb::import_report evsmry " << ( evsmry ? evsmry->sstr() : "-" ) << "\n" ;

    bool skip_import = is_imported(run);


    if(skip_import)
    {
        std::cout <<  "-sreportdb::import_report - SKIP AS ALREADY IMPORTED\n";
    }
    else
    {

        int64_t versionset_id = _import_versionset(run);

        int64_t run_id        = _import_run(_fold, run, evsmry, versionset_id, metadata );

        int64_t last_evt_id   = _import_evsmry( evsmry, run_id );

        std::cout
            << "-sreportdb::import_report"
            << " versionset_id " << versionset_id
            << " run_id " << run_id
            << " last_evt_id " << last_evt_id
            << "\n"
            ;
    }

    std::cout
        << "]sreportdb::import_report "
        << ( _fold ? _fold : "-" )
        << " skip_import " << ( skip_import ? "YES" : "NO ")
        << "\n\n"
        ;

    return 0 ;
}


inline bool sreportdb::is_imported( const NP* run )
{
    //std::string ci_pipeline_source = run->get_meta<std::string>("CI_PIPELINE_SOURCE","");
    int64_t ci_pipeline_id         = run->get_meta<int64_t>("CI_PIPELINE_ID",-1);
    int64_t ci_job_id              = run->get_meta<int64_t>("CI_JOB_ID",-1);
    int64_t run_timestamp          = run->get_meta<int64_t>("InitTimestamp",0);   // formerly _init_stamp


    bool already = is_imported_combi(ci_pipeline_id, ci_job_id, run_timestamp );
    //bool already = ci_pipeline_id > -1 ? is_imported_ci(ci_pipeline_id, ci_job_id) : is_imported_nonci( run_timestamp );

    return already ;
}


/**
sreportdb::is_imported_combi
-----------------------------

Combi approach avoids separate methods for ci and nonci but with
the expense of more complicated sql.

**/

inline bool sreportdb::is_imported_combi( int64_t ci_pipeline_id, int64_t ci_job_id, int64_t run_timestamp )
{
    // A single query that adapts seamlessly based on the value of ci_pipeline_id
    const char* check_sql =
        "SELECT EXISTS("
        "    SELECT 1 FROM opticks_runs "
        "    WHERE (? != -1 AND ci_pipeline_id = ? AND ci_job_id = ?) " // CI path
        "       OR (? = -1  AND run_timestamp = ?)"                    // Local path
        ");";

    NSQLiteStmt selector(db->db, check_sql);

    // Bind the variables matching the order of the '?' placeholders
    auto _result = selector.query_scalar<int64_t>(
        ci_pipeline_id, ci_pipeline_id, ci_job_id, // CI parameters
        ci_pipeline_id, run_timestamp              // Local parameters
    );

    int64_t result = _result.has_value() ? _result.value() : -1 ;
    bool already = (result == 1);

    std::cout << "-sreportdb::is_imported_ "
              << " ci_pipeline_id " << ci_pipeline_id
              << " ci_job_id " << ci_job_id
              << " run_timestamp " << run_timestamp
              << " result " << result << "\n" ;

    return already ;
}

inline bool sreportdb::is_imported_nonci( int64_t run_timestamp )
{
    const char* check_sql =
        "SELECT EXISTS("
        "    SELECT 1 FROM opticks_runs "
        "    WHERE ( run_timestamp = ? ) "
        ");";

    NSQLiteStmt selector(db->db, check_sql);

    auto _result = selector.query_scalar<int64_t>( run_timestamp );
    int64_t result = _result.has_value() ? _result.value() : -1 ;
    bool already = (result == 1);

    std::cout << "-sreportdb::is_imported_nonci "
              << " run_timestamp " << run_timestamp
              << " result " << result << "\n" ;

    return already ;
}

inline bool sreportdb::is_imported_ci( int64_t ci_pipeline_id, int64_t ci_job_id )
{
    const char* check_sql =
        "SELECT EXISTS("
        "    SELECT 1 FROM opticks_runs "
        "    WHERE ( ci_pipeline_id = ? AND ci_job_id = ? ) "
        ");";

    NSQLiteStmt selector(db->db, check_sql);

    auto _result = selector.query_scalar<int64_t>( ci_pipeline_id, ci_job_id );
    int64_t result = _result.has_value() ? _result.value() : -1 ;
    bool already = (result == 1);

    std::cout << "-sreportdb::is_imported_ci "
              << " ci_pipeline_id " << ci_pipeline_id
              << " ci_job_id " << ci_job_id
              << " result " << result << "\n"
              ;

    return already ;
}




/**
sreportdb::_import_versionset
------------------------------

Invoked from sreportdb::import_report

1. inserts or ignores version information obtained from run array metadata into the versionset table,
   the "or ignore" means that a row is added to the table only for a new versionset
2. queries the versionset table to yield versionset_id
3. returns the versionset id


The below script is an example of using inplace sed editing to add a line
to multiple run_meta.txt in report archive as the "gitlab-runner" user
(SREPORT_ARCHIVE_DIR: /data1/gitlab-runner/sreport_archive)::


    #!/bin/bash
    ## sudo -u gitlab-runner bash -c "$PWD/add_Custom4Version_line.sh"
    find . -name "run_meta.txt"
    find . -name "run_meta.txt" -exec sed -i '/Geant4/a Custom4Version:108' {} +

**/


inline int64_t sreportdb::_import_versionset(const NP* run)
{
    std::cout << "[sreportdb::import_versionset run " << ( run ? run->sstr() : "-" ) << "\n";
    int64_t opticks_version    = run->get_meta<int64_t>("OpticksVersion",0);
    int64_t geant4_version     = run->get_meta<int64_t>("Geant4Version",0);
    int64_t custom4_version    = run->get_meta<int64_t>("Custom4Version",0);
    int64_t optix_version      = run->get_meta<int64_t>("OptiXVersion",0);
    int64_t compute_capability = run->get_meta<int64_t>("ComputeCapability",0);
    int64_t cuda_version       = run->get_meta<int64_t>("CUDAVersion",0);
    int64_t cuda_driver        = run->get_meta<int64_t>("CUDADriver",0);
    std::string nvidia_driver  = run->get_meta<std::string>("NvidiaDriverVersion", "");

   const char* insert_or_ignore_sql =
        "INSERT OR IGNORE INTO opticks_versionset ("
        "opticks_version, geant4_version, custom4_version, optix_version, "
        "compute_capability, "
        "cuda_version, cuda_driver, nvidia_driver "
        ") VALUES ("
        "?,?,?,?,"
        "?,"
        "?,?,?"
        ");";

    NSQLiteStmt insertor(db->db, insert_or_ignore_sql);
    bool insertor_success = insertor.execute(
                      opticks_version, geant4_version, custom4_version, optix_version,
                      compute_capability,
                      cuda_version, cuda_driver, nvidia_driver
                     );
    assert(insertor_success);


    // 3. Query the ID (Whether newly created or pre-existing)
    const char* select_sql =
        "SELECT id FROM opticks_versionset WHERE"
        " opticks_version    = ? AND"
        " geant4_version     = ? AND"
        " custom4_version    = ? AND"
        " optix_version      = ? AND"
        " compute_capability = ? AND"
        " cuda_version       = ? AND"
        " cuda_driver        = ? AND"
        " nvidia_driver      = ?"
        ";"
        ;

    NSQLiteStmt selector(db->db, select_sql);

    auto result = selector.query_scalar<int64_t>(
                      opticks_version, geant4_version, custom4_version, optix_version,
                      compute_capability,
                      cuda_version, cuda_driver, nvidia_driver
                      );


    int64_t versionset_id = result.has_value() ? result.value() : -1 ;
    if(versionset_id < 0) throw std::runtime_error("sreportdb::_import_versionset FAILED - versionset could not be verified or created.");
    std::cout << "]sreportdb::import_versionset versionset_id " << versionset_id << "\n";
    return versionset_id ;
}

/**
sreportdb::_import_run
-----------------------

Invoked from sreportdb::import_report
Inserts inserts into opticks_runs table metadata obtained from:

1. run array metadata
2. evsmry array metadata initialization times
3. fold and versionset_id from arguments

Returns run_id of the single added row.

    // HMM: MAYBE A JSON BLOB TOO - FOR CHANGE ISOLATION



**/

inline int64_t sreportdb::_import_run(const char* _fold, const NP* run, const NP* evsmry, int64_t versionset_id, const std::string& metadata )
{
    std::cout
       << "[sreportdb::_import_run"
       << " versionset_id " << versionset_id
       << " metadata " << metadata
       << "\n"
       ;

    assert( versionset_id > 0 );

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

    // OR IGNORE
    const char* sql = "INSERT INTO opticks_runs ("
                      "versionset_id, run_timestamp,"
                      "fold, script, script_arg, executable,"
                      "test, gpu, geometry, tree_digest,"
                      "max_bounce, dt_geometry_load, dt_geometry_upload, total_events,"
                      "ci_pipeline_source, ci_pipeline_id, ci_job_id,"
                      "metadata"
                      ") "
                      "VALUES ("
                      "?,?,"
                      "?, ?, ?, ?, "
                      "?, ?, ?, ?, "
                      "?, ?, ?, ?, "
                      "?, ?, ?, "
                      "?"
                      ");";

    // NULLIF in the SQL the fallback ci values yields NULL in the table
    // "NULLIF(?,''), NULLIF(?,-1), NULLIF(?,-1)"

    NSQLiteStmt inserter(db->db, sql);
    bool success = inserter.execute(
                      versionset_id, run_timestamp,
                      fold, script, script_arg, executable,
                      test, gpu, geometry, tree_digest,
                      max_bounce, dt_geometry_load, dt_geometry_upload, total_events,
                      ci_pipeline_source, ci_pipeline_id, ci_job_id,
                      metadata);

    assert(success);

    int64_t run_id = sqlite3_last_insert_rowid(db->db);
    std::cout << "]sreportdb::_import_run run_id " << run_id << "\n";
    return run_id ;
}

inline std::string sreportdb::desc_runs() const
{
    const char* desc_sql = "SELECT id, run_timestamp, ci_pipeline_id, ci_job_id FROM opticks_runs;";
    NSQLiteStmt desc(db->db, desc_sql);
    std::stringstream ss;

    int64_t id = 0;
    int64_t run_timestamp = 0;
    int64_t ci_pipeline_id = 0;
    int64_t ci_job_id = 0;

    // Beautifully elegant unpacking directly into variables
    while (desc.fetch(id, run_timestamp, ci_pipeline_id, ci_job_id))
    {
        ss << "(opticks_runs)"
           << " id: "             << std::setw(10) << id
           << " run_timestamp: "  << std::setw(16) << run_timestamp
           << " ci_pipeline_id: " << std::setw(10) << (ci_pipeline_id == -1 ? "LOCAL" : std::to_string(ci_pipeline_id))
           << " ci_job_id: "      << std::setw(10) << (ci_job_id      == -1 ? "LOCAL" : std::to_string(ci_job_id)     )
           << "\n";
    }
    std::string str = ss.str() ;
    return str ;
}

/**
sreportdb::_import_evsmry
--------------------------

Invoked from sreportdb::import_report

Inserts one row for each event with counts and timings plus
the run_id reference from the argument. Returns the last_evt_id
from the last row added.

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


