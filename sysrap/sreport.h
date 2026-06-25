#pragma once
/**
sreport.h
==========




**/

#include "NPFold.h"
#include "sfilesystem.h"

#define WITH_SUBMETA 1

struct sreport
{
    static constexpr const char* COLLECTION_PREFIX = "sreport_" ;
    static constexpr const char* JUNCTURE = "SEvt__Init_RUN_META,SEvt__BeginOfRun,SEvt__EndOfRun,SEvt__Init_RUN_META" ;
    static constexpr const char* RANGES = R"(
        SEvt__Init_RUN_META:CSGFoundry__Load_HEAD                     ## init
        CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL                   ## load_geom
        CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL                   ## upload_geom
        A%0.3d_QSim__simulate_HEAD:A%0.3d_QSim__simulate_LBEG         ## slice_genstep
        A%0.3d_QSim__simulate_PRUP:A%0.3d_QSim__simulate_PREL         ## upload genstep slice
        A%0.3d_QSim__simulate_PREL:A%0.3d_QSim__simulate_POST         ## simulate slice
        A%0.3d_QSim__simulate_POST:A%0.3d_QSim__simulate_DOWN         ## download slice
        A%0.3d_QSim__simulate_LEND:A%0.3d_QSim__simulate_PCAT         ## concat slices
        A%0.3d_QSim__simulate_BRES:A%0.3d_QSim__simulate_TAIL         ## save arrays
       )" ;

    static constexpr const char* sreport__LEVEL = "sreport__LEVEL" ;
    static constexpr const char* sreport__VERBOSE = "sreport__VERBOSE" ;
    static constexpr const char* sreport__CONFIG = "sreport__CONFIG" ;
    int     LEVEL ;
    bool    VERBOSE ;
    const char* CONFIG ;

    NP*       run ;       // dummy array that exists just to hold metadata
    NP*       runprof ;   // "prof" would be a better name now : but awkward to change
    NP*       ranges ;    // detailed range timing extracted from SProf.txt

    NPFold*   substamp ;
    NPFold*   subprofile ;
    NPFold*   submeta ;   // folder of arrays of metadata values extracted from folder metadata - NOW ONLY RELEVANT FOR NON-PROFILE INFO
    NPFold*   submeta_NumPhotonCollected ; // RE-USE of submeta machinery with a single column - HMM THIS NEEDS TO BE REPLACED WITH SProf ANNOTATIONS
    NPFold*   subcount ;

    sreport();

    NPFold* serialize() const ;
    NPFold* serialize_copy() const ;
    void    import( const NPFold* fold );
    void save(const char* dir) const ;

    void save_into_archive(const char* archive_dir) const ;

    static long long   FindIndexOfMaxIndexedDirname(const char* collection_dir);
    static std::string FormIndexedDirname(long long index);
    static bool        IsIndexedDirname(const char* dirname);

    static sreport* Load(const char* dir) ;

    bool        is_config(const char* method, char delim=',') const;
    static bool IsConfig(const char* config, const char* label, char delim=',');

    std::string desc() const ;
    std::string desc_run() const ;
    std::string desc_runprof() const ;
    std::string desc_ranges() const ;
    std::string desc_substamp() const ;
    std::string desc_subprofile() const ;
    std::string desc_submeta() const ;
    std::string desc_subcount() const ;
};

inline sreport::sreport()
    :
    LEVEL(U::GetEnvInt(sreport__LEVEL,0)),
    VERBOSE(getenv(sreport__VERBOSE) != nullptr),
    CONFIG(U::GetEnv(sreport__CONFIG,"")),
    run( nullptr ),
    runprof( nullptr ),
    ranges( nullptr ),
    substamp( nullptr),
    subprofile( nullptr ),
    submeta( nullptr ),
    submeta_NumPhotonCollected( nullptr ),
    subcount( nullptr )
{
}
inline NPFold* sreport::serialize() const
{
    NPFold* smry = new NPFold ;
    smry->add("run", run ) ;
    smry->add("runprof", runprof ) ;
    smry->add("ranges", ranges ) ;
    smry->add_subfold("substamp", substamp ) ;
    smry->add_subfold("subprofile", subprofile ) ;
    smry->add_subfold("submeta", submeta ) ;
    smry->add_subfold("submeta_NumPhotonCollected", submeta_NumPhotonCollected ) ;
    smry->add_subfold("subcount", subcount ) ;
    return smry ;
}

inline NPFold* sreport::serialize_copy() const
{
    NPFold* smry = new NPFold ;
    smry->add("run", run->copy() ) ;
    smry->add("runprof", runprof->copy() ) ;
    smry->add("ranges", ranges->copy() ) ;

    smry->add_subfold("substamp", substamp->deepcopy() ) ;
    smry->add_subfold("subprofile", subprofile->deepcopy() ) ;
    smry->add_subfold("submeta", submeta->deepcopy() ) ;
    smry->add_subfold("submeta_NumPhotonCollected", submeta_NumPhotonCollected->deepcopy() ) ;
    smry->add_subfold("subcount", subcount->deepcopy() ) ;
    return smry ;
}


inline void sreport::import(const NPFold* smry)
{
    run = smry->get("run")->copy() ;
    runprof = smry->get("runprof")->copy() ;
    ranges = smry->get("ranges")->copy() ;
    substamp = smry->get_subfold("substamp");
    subprofile = smry->get_subfold("subprofile");
    submeta = smry->get_subfold("submeta");
    submeta_NumPhotonCollected = smry->get_subfold("submeta_NumPhotonCollected");
    subcount = smry->get_subfold("subcount");
}
inline void sreport::save(const char* dir) const
{
    NPFold* smry = serialize();
    smry->save_verbose(dir);
}

/**
sreport::save_into_archive
----------------------------

Saves report into a collection of indexed reports using
an index based on the existing indices found in the *collection_dir*
for example::

     collection_dir/sreport_0000000/
     collection_dir/sreport_0000001/
     collection_dir/sreport_0000101/

**/

inline void sreport::save_into_archive(const char* archive_dir) const
{
    NPFold* smry = serialize_copy();
    long long max_index = FindIndexOfMaxIndexedDirname(archive_dir);
    long long next_index = max_index + 1 ; 
    std::string indexed_dirname = FormIndexedDirname(next_index);
    smry->save(archive_dir, indexed_dirname.c_str() );
}


inline long long sreport::FindIndexOfMaxIndexedDirname(const char* collection_dir)
{
    return sfilesystem::find_index_of_max_indexed_dirname(collection_dir, COLLECTION_PREFIX);
}
inline std::string sreport::FormIndexedDirname(long long index)
{
    return sfilesystem::form_indexed_dirname(index, COLLECTION_PREFIX);
}
inline bool sreport::IsIndexedDirname(const char* dirname)  // static
{
    return sfilesystem::is_indexed_dirname( dirname, COLLECTION_PREFIX );  
}



inline sreport* sreport::Load(const char* dir) // static
{
    NPFold* smry = NPFold::Load(dir) ;
    sreport* report = new sreport ;
    report->import(smry) ;
    return report ;
}

/**
sreport::is_config
---------------------

Returns true when CONFIG is blank OR the *label* is present in the CONFIG.
This allows the output to be controlled eg::

**/
inline bool sreport::is_config(const char* label, char delim) const
{
    return IsConfig(CONFIG, label, delim);
}

inline bool sreport::IsConfig(const char* config, const char* label, char delim) // static
{
    if( config == nullptr || 0 == strcmp(config,"") || label == nullptr ) return true ;

    std::string target = label ;

    std::vector<std::string> tokens;
    std::stringstream ss(config);
    std::string token;
    while (std::getline(ss, token, delim)) tokens.push_back(token);

    auto it = std::find(tokens.begin(), tokens.end(), target);
    bool in_config = it != tokens.end() ;
    return in_config ;
}

/**
sreport::desc
---------------

Control what to include with::

    export sreport__CONFIG=run,runprof,ranges,substamp,submeta,subcount
    export sreport__CONFIG=substamp


OR::

    sreport__CONFIG=substamp sreport

From cxs_min.sh level::

    sreport__CONFIG=substamp cxs_min.sh report

**/


inline std::string sreport::desc() const
{
    std::stringstream ss ;
    ss << "[sreport.desc CONFIG[" << ( CONFIG ? CONFIG : "-" ) << "]\n"
       << ( is_config("run")      ? desc_run() : "" )
       << ( is_config("runprof")  ? desc_runprof() : "" )
       << ( is_config("ranges")   ? desc_ranges()  : "" )
       << ( is_config("substamp") ? desc_substamp() : "" )
       << ( is_config("submeta")  ? desc_submeta()  : "" )
       << ( is_config("subcount") ? desc_subcount() : "" )
       << "]sreport.desc" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sreport::desc_run() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_run (run is dummy small array used as somewhere to hang metadata) "
       << ( run ? run->sstr() : "-" )
       << "\n"
       << "[sreport.desc_run.descMetaKVS " << std::endl
       << ( run ? run->descMetaKVS(JUNCTURE, RANGES) : "-" ) << std::endl
       << "]sreport.desc_run.descMetaKVS " << std::endl
       << "]sreport.desc_run" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}


inline std::string sreport::desc_runprof() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_prof" << std::endl
       << ( runprof ? runprof->sstr() : "-" ) << std::endl
       << ".sreport.desc_runprof.descTable " << std::endl
       << ( runprof ? runprof->descTable<int64_t>(17) : "-" ) << std::endl
       << "]sreport.desc_runprof" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

/**
sreport::desc_ranges
---------------------

::

    sreport.desc_ranges ranges : (11, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {} (11, 5, )
                                                      ta                tb                ab                ia                ib
                                  init          0.000000          0.006884              6884                 0                 2          0.006884                              init
                             load_geom          0.006884          1.169323           1162439                 2                 3          1.162439                         load_geom
                           upload_geom          1.169364          1.557288            387924                 4                 5          0.387924                       upload_geom
                         slice_genstep          1.557645          1.557688                43                 9                10          0.000043                     slice_genstep
                  upload genstep slice          1.557695          1.571605             13910                11                12          0.013910              upload genstep slice
                        simulate slice          1.571605          1.868877            297272                12                13          0.297272                    simulate slice
                        download slice          1.868877          1.887110             18233                13                14          0.018233                    download slice
                         slice_genstep          1.953854          1.953877                23                21                22          0.000023                     slice_genstep
                  upload genstep slice          1.953884          1.954766               882                23                24          0.000882              upload genstep slice
                        simulate slice          1.954766          2.247558            292792                24                25          0.292792                    simulate slice
                        download slice          2.247558          2.268320             20762                25                26          0.020762                    download slice
    num_timestamp 22 auto-offset from t0 1782097507944130
                                TOTAL:          15842132          18043296           2201164               144               156

                                  init : SEvt__Init_RUN_META:CSGFoundry__Load_HEAD:init
                             load_geom : CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL:load_geom
                           upload_geom : CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL:upload_geom
                         slice_genstep : A000_QSim__simulate_HEAD:A000_QSim__simulate_LBEG:slice_genstep
                  upload genstep slice : A000_QSim__simulate_PRUP:A000_QSim__simulate_PREL:upload genstep slice
                        simulate slice : A000_QSim__simulate_PREL:A000_QSim__simulate_POST:simulate slice
                        download slice : A000_QSim__simulate_POST:A000_QSim__simulate_DOWN:download slice
                         slice_genstep : A001_QSim__simulate_HEAD:A001_QSim__simulate_LBEG:slice_genstep
                  upload genstep slice : A001_QSim__simulate_PRUP:A001_QSim__simulate_PREL:upload genstep slice
                        simulate slice : A001_QSim__simulate_PREL:A001_QSim__simulate_POST:simulate slice
                        download slice : A001_QSim__simulate_POST:A001_QSim__simulate_DOWN:download slice
    ]NP::descTable_ {} (11, 5, )


Repetitions in ia/ib indices and ta/tb values are normal, that just shows the end of one tag range
is the same as the start of the next range.

**/


inline std::string sreport::desc_ranges() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_ranges"
       << " ranges : " << ( ranges ? ranges->sstr() : "-" ) << std::endl
       << ".sreport.desc_ranges.descTable "
       << " ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )"
       << "\n"
       << ( ranges ? ranges->descTable<int64_t>(17) : "-" ) << std::endl
       << "]sreport.desc_ranges" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

/**
report::desc_substamp
-----------------------

NB substamp is only useful for debugging (not production)
as it relies on SEvt metadata - with are not saved in production
as far too big.


The substamp NPFold is created by::

   substamp = fold->subfold_summary("substamp",   ASEL, BSEL)


**/

inline std::string sreport::desc_substamp() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_substamp" << std::endl
       ;
    if(VERBOSE) ss
       << "[sreport.desc_substamp.VERBOSE" << std::endl
       << ( substamp ? substamp->desc() : "-" )
       << "]sreport.desc_substamp.VERBOSE" << std::endl
       ;

    ss << "[sreport.desc_substamp.compare_subarrays_report" << std::endl
       <<  ( substamp ? substamp->compare_subarrays_report<double, int64_t>( "delta_substamp", "a", "b" ) : "-" )
       << "]sreport.desc_substamp.compare_subarrays_report" << std::endl
       << "]sreport.desc_substamp" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}


inline std::string sreport::desc_subprofile() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_subprofile" << std::endl
       ;
    if(VERBOSE) ss
       << "[sreport.desc_subprofile.VERBOSE" << std::endl
       << ( subprofile ? subprofile->desc() : "-" )
       << "]sreport.desc_subprofile.VERBOSE" << std::endl
       ;

    /*
    ss << "[sreport.desc_subprofile.compare_subarrays_report" << std::endl
       <<  ( subprofile ? subprofile->compare_subarrays_report<double, int64_t>( "delta_subprofile", "a", "b" ) : "-" )
       << "]sreport.desc_subprofile.compare_subarrays_report" << std::endl
    */

    ss
       << "]sreport.desc_subprofile" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}


/**
sreport::desc_submeta - FOR PROFILING INFO NOW REPLACED WITH SProf.txt BASED APPROACHES
-------------------------------------------------------------------------------------------

TODO: General NPFold::desc is acting as placeholder here - replace it with something more informative

::

    sreport__CONFIG=submeta cxs_min.sh report


HMM int64_t can hold everything other than t_Launch::

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first]$ cat A000/NPFold_meta.txt
    NumPhotonCollected:1000000
    NumGenstepCollected:1
    MaxBounce:31
    site:SEvt::endMeta
    hitmask:8192
    index:0
    instance:0
    SEvt__beginOfEvent_0:1782097509501642,8128336,1264128
    SEvt__beginOfEvent_1:1782097509501767,8128336,1264128
    SEvt__endOfEvent_0:1782097509831333,15481740,1283840
    t_BeginOfEvent:1782097509501655
    t_setGenstep_0:0
    t_setGenstep_1:0
    t_setGenstep_2:0
    t_setGenstep_3:1782097509501835
    t_setGenstep_4:1782097509502034
    t_setGenstep_5:1782097509502050
    t_setGenstep_6:1782097509502068
    t_setGenstep_7:1782097509503254
    t_setGenstep_8:1782097509515733
    t_PreLaunch:1782097509515780
    t_PostLaunch:1782097509813007
    t_EndOfEvent:1782097509831341
    t_Event:329686
    t_Launch:0.297208


**/


inline std::string sreport::desc_submeta() const
{
    const NP* a = submeta ? submeta->get("a") : nullptr ;
    const NP* b = submeta ? submeta->get("b") : nullptr ;

    std::stringstream ss ;
    ss << "[sreport.desc_submeta {submeta for profiling info is replaced by *ranges* which just needs SProf.txt - not SEvt NPFold_meta.txt} " << std::endl
       //<< ( submeta ? submeta->desc() : "-" )
       << "\n"
       << " a " << ( a ? a->sstr() : "-" ) << "\n"
       << " b " << ( b ? b->sstr() : "-" ) << "\n"
       << ( a && a->num_dim() == 2 ? a->descTable<int64_t>(9,"submeta.a") : "-" ) << "\n"
       << ( b && b->num_dim() == 2 ? b->descTable<int64_t>(9,"submeta.b") : "-" ) << "\n"
       << "]sreport.desc_submeta" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sreport::desc_subcount() const
{
    const NP* a = subcount ? subcount->get("a") : nullptr ;
    const NP* b = subcount ? subcount->get("b") : nullptr ;

    std::stringstream ss ;
    ss << "[sreport.desc_subcount\n"
       //<< ( subcount ? subcount->desc() : "-" ) << "\n"
       //<< " a " << ( a ? a->sstr() : "-" ) << "\n"
       //<< " b " << ( b ? b->sstr() : "-" ) << "\n"
       << ( a && a->num_dim() == 2 ? a->descTable<int32_t>(9,"subcount.a") : "-" ) << "\n"
       << ( b && b->num_dim() == 2 ? b->descTable<int32_t>(9,"subcount.b") : "-" ) << "\n"
       << "]sreport.desc_subcount\n"
       ;
    std::string str = ss.str() ;
    return str ;
}


