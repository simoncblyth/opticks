/**
sreport.cc : Summarize + Present SEvt/NPFold metadata time stamps
=============================================================================

::

    ~/opticks/sysrap/tests/sreport.sh
    ~/opticks/sysrap/tests/sreport.sh grab
    ~/opticks/sysrap/tests/sreport.sh ana


Summarizes SEvt/NPFold metadata time stamps into substamp arrays
grouped by NPFold path prefix. The summary NPFold is presented textually
and saved to allow plotting from python.

+-----+---------------------------------+-------------------------+
| key | SEvt/NPFold path prefix         |  SEvt type              |
+=====+=================================+=========================+
| a   | "//A" eg: //A000 //A001         | Opticks/QSim SEvt       |
+-----+---------------------------------+-------------------------+
| b   | "//B" eg: //B000 //B001         | Geant4/U4Recorder SEvt  |
+-----+---------------------------------+-------------------------+

The tables are presented with row and column labels and the
summary NPFold is saved to DIR_sreport sibling to invoking DIR
which needs to contain SEvt/NPFold folders corresponding to the path prefix.
The use of NPFold::LoadNoData means that only SEvt NPFold/NP
metadata is loaded. Excluding the array data makes the load
very fast and able to handle large numbers of persisted SEvt NPFold.

Usage from source "run" directory creates the report saving into eg ../ALL3_sreport::

    epsilon:~ blyth$ cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/G4CXTest/ALL3
    epsilon:ALL3 blyth$ sreport
    epsilon:ALL3 blyth$ ls -alst ../ALL3_sreport

Usage from report directory loads and presents the report::

    epsilon:ALL3 blyth$ cd ../ALL3_sreport/
    epsilon:ALL3_sreport blyth$ sreport

Note that this means that can rsync just the small report directory
and still be able to present the report and make plots on laptop concerning
run folders with many large arrays left on the server.


Debugging Notes
-----------------

Debugging this is whacky as its mostly stringstream preparation
so cout/cerr reporting sometimes seems out of place compared to
the report output. For this reason its important to label most
output with where it comes from to speedup understanding+debug.

**/

#include "NPFold.h"

#define WITH_SUBMETA 1

struct sreport
{
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
    void    import( const NPFold* fold );
    void save(const char* dir) const ;
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



/**
sreport_Creator
---------------

1. loads folder metadata with NPFold::LoadNoData
2. instanciates and populates sreport instance

**/

struct sreport_Creator
{
    static constexpr const char* ASEL = "a://A" ;
    static constexpr const char* BSEL = "b://B" ;

    bool VERBOSE ;
    const char* dirp ;
    const char* CONFIG ;
    bool _creator ;

    NPFold*    fold ;
    bool fold_valid ;
    const NP*  run ;
    sreport*   report ;

    sreport_Creator(  const char* dirp_, const char* CONFIG);
    void init();
    void init_runprof_run_ranges_from_SProf();
    void init_substamp();
    void init_subprofile();
    void init_submeta();
    void init_subcount();

    std::string desc() const ;
    std::string desc_fold() const ;
    std::string desc_fold_detail() const ;
    std::string desc_run() const ;
};

inline sreport_Creator::sreport_Creator( const char* dirp_, const char* _CONFIG )
    :
    VERBOSE(getenv("sreport_Creator__VERBOSE") != nullptr),
    dirp(dirp_ ? strdup(dirp_) : nullptr),
    CONFIG(_CONFIG),
    _creator(sreport::IsConfig(CONFIG, "creator")),
    fold(NPFold::LoadNoData(dirp)),
    fold_valid(NPFold::IsValid(fold)),
    run(fold_valid ? fold->get("run") : nullptr),
    report(new sreport)
{
    if(_creator) std::cout
        << "[sreport_Creator::sreport_Creator"
        << " fold_valid " << ( fold_valid ? "YES" : "NO " )
        << " run " << ( run ? "YES" : "NO " )
        << "\n"
        ;

    init();

    if(_creator) std::cout << "]sreport_Creator::sreport_Creator" << std::endl ;
}


/**
sreport_Creator::init
-----------------------

1. construct SProf derived metadata arrays with init_runprof_run_ranges_from_SProf
2. construct subfold derived arrays and fold using the *sub* methods


Q: Are *sub* methods still relevant now that have init_runprof_run_ranges_from_SProf ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *sub* methods have a fundamental disadvantage over the SProf.txt based approach
in that they require SEvt saving - so they are limited to tests with debug arrays enabled.
This contrasts to the SProf approach which can work in production with only SProf.txt saved.

However the *sub* methods provide richer information that SProf currently does.
Although some of that *sub* functionality could be adapted to working from SProf.txt
data, drawing on the *sub* methods.


Q: Recall that SProf.txt includes annotations ? Are those yet used from sreport ? A: NO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SProf.hh writes and reads the annotations into META vector of strings that are mostly blanks::

    A000_QSim__simulate_DOWN:1782097509831240,15481740,1283840
    A000_QSim__simulate_LEND:1782097509831276,15481740,1283840 # slice=1,max_slot_M=100
    A000_QSim__simulate_PCAT:1782097509831298,15481740,1283840
    A000_QSim__simulate_BRES:1782097509831315,15481740,1283840 # numGenstepCollected=1,numPhotonCollected=1000000,numHit=213947
    A000_QSim__reset_HEAD:1782097509831323,15481740,1283840


**/


inline void sreport_Creator::init()
{
    if(_creator) std::cout << "[sreport_Creator::init\n" ;

    init_runprof_run_ranges_from_SProf();

    init_substamp();
    init_subprofile();
    init_submeta();
    init_subcount();

    if(_creator) std::cout << "]sreport_Creator::init\n" ;
}

/**
sreport_Creator::init_runprof_run_ranges_from_SProf
----------------------------------------------------

Creates three arrays:

runprof
    coarse event timings from SProf.txt, effectively "grep Index SProf.txt"
run
    dummy array that exists just to hold run_meta.txt
ranges
    detailed profile timings table obtained from SProf.txt



The SProf has the advantage of almost always being available, as the SProf.txt is small
unlike the full arrays.

1. read SProf.txt into meta string
2. create report->runprof array from the "Index" lines ("setIndex","endIndex") shaped (2,3) for the below example
3. create report->run (dummy array with run metadata)
4. create report->ranges. eg shaped (8,5) with the below example : this included time deltas between the keys
   that match the wildcard resolved sreport::RANGES

Analysis of the SProf.txt written by SProf.hh, eg::

    A[blyth@localhost ALL1_Debug_Philox_ref1]$ cat SProf.txt
    SEvt__Init_RUN_META:1760707884870593,46464,8064
    CSGOptiX__SimulateMain_HEAD:1760707884871147,46464,10752
    CSGFoundry__Load_HEAD:1760707884871171,46464,11200
    CSGFoundry__Load_TAIL:1760707885851123,5419968,883988
    CSGOptiX__Create_HEAD:1760707885851159,5419968,883988
    CSGOptiX__Create_TAIL:1760707886286856,7316444,1222084
    A000_QSim__simulate_HEAD:1760707886286900,7316444,1222084
    A000_SEvt__BeginOfRun:1760707886286914,7316444,1222084
    A000_SEvt__beginOfEvent_FIRST_EGPU:1760707886287034,7316444,1222084
    A000_SEvt__setIndex:1760707886287057,7316444,1222084
    A000_QSim__simulate_LBEG:1760707886287202,7316444,1222084
    A000_QSim__simulate_PRUP:1760707886287207,7316444,1222084
    A000_QSim__simulate_PREL:1760707886288393,8266716,1222980
    A000_QSim__simulate_POST:1760707886441594,8266716,1227908
    A000_QSim__simulate_DOWN:1760707886541324,8373000,1334844
    A000_QSim__simulate_LEND:1760707886541353,8373000,1334844
    A000_QSim__simulate_PCAT:1760707886541381,8373000,1334844
    A000_QSim__simulate_BRES:1760707886541433,8373000,1334844 # numGenstepCollected=10,numPhotonCollected=1000000,numHit=200397
    A000_QSim__reset_HEAD:1760707886541441,8373000,1334844
    A000_SEvt__endIndex:1760707886541457,8373000,1334844
    A000_SEvt__EndOfRun:1760707887055687,8266716,1229000
    A000_QSim__reset_TAIL:1760707887055757,8266716,1229000
    A000_QSim__simulate_TAIL:1760707887055768,8266716,1229000
    CSGOptiX__SimulateMain_TAIL:1760707887056235,8266716,1229000
    A[blyth@localhost ALL1_Debug_Philox_ref1]$


Grepping Index lines from SProf.txt gives timing overview::

    A[blyth@localhost ALL1_Debug_Philox_medium_scan_first]$ grep Index SProf.txt
    A000_SEvt__setIndex:1782097509501661,8128336,1264128
    A000_SEvt__endIndex:1782097509831341,15481740,1283840
    A001_SEvt__setIndex:1782097509897933,15468368,1271048
    A001_SEvt__endIndex:1782097510212520,15481680,1284040


**/

inline void sreport_Creator::init_runprof_run_ranges_from_SProf()
{
    if(_creator) std::cout << "[sreport_Creator::init_runprof_run_ranges_from_SProf\n" ;

    std::string meta = U::ReadString2_("SProf.txt");

    report->runprof = NP::MakeMetaKVProfileArray(meta, "Index") ;
    if(_creator) std::cout << "-sreport_Creator::init.SProf:runprof   :" << ( report->runprof ? report->runprof->sstr() : "-" ) << " {runprof effectively: grep Index SProf.txt}" << std::endl ;

    report->run     = run ? run->copy() : nullptr ;
    if(_creator) std::cout << "-sreport_Creator::init_SProf.run       :" << ( report->run ? report->run->sstr() : "-" ) << " {run is dummy array just for holding metadata} " << std::endl ;

    report->ranges = run ? NP::MakeMetaKVS_ranges2( meta, sreport::RANGES ) : nullptr ;
    if(_creator) std::cout << "-sreport_Creator::init_SProf.ranges2   :" << ( report->ranges ?  report->ranges->sstr() : "-" ) << " {ranges provides detailed timing info extracted from SProf.txt}" <<  std::endl ;

    if(_creator) std::cout << "]sreport_Creator::init_runprof_run_ranges_from_SProf\n" ;
}

/**
sreport_Creator::init_substamp
-------------------------------

1. create report->substamp from stamps found in NPFold_meta.txt of the specified subfolders

**/


inline void sreport_Creator::init_substamp()
{
    if(_creator) std::cout << "[sreport_Creator::init_substamp\n" ;
    if(_creator) std::cout << "-sreport_Creator::init_substamp fold_valid " << ( fold_valid ? "Y" : "N" ) << std::endl ;

    report->substamp   = fold_valid ? fold->subfold_summary("substamp",   ASEL, BSEL) : nullptr ;

    if(_creator) std::cout << "-sreport_Creator::init_substamp ((NPFold)report.substamp).stats [" << ( report->substamp ? report->substamp->stats() : "-" ) << "]\n" ;
    if(_creator) std::cout << "]sreport_Creator::init_substamp\n" ;
}

/**
sreport_Creator::init_subprofile
----------------------------------

1. create report->subprofile from profile triplet metadata from the NPFold_meta.txt of the specified subfolders

**/

inline void sreport_Creator::init_subprofile()
{
    if(_creator) std::cout << "[sreport_Creator::init_subprofile\n" ;

    report->subprofile = fold_valid ? fold->subfold_summary("subprofile", ASEL, BSEL) : nullptr ;

    if(_creator) std::cout << "-sreport_Creator::init_subprofile :[" << ( report->subprofile ? report->subprofile->stats() : "-" )  << "]\n" ;
    if(_creator) std::cout << "]sreport_Creator::init_subprofile\n" ;
}

/**
sreport_Creator::init_submeta
-------------------------------

1. create report->submeta from subfold metadata
2. report->submeta_NumPhotonCollected from subfold metadata

This consolidates metadata from multiple persisted SEvt specified by "//A" OR "//B"
corresponding to "/A000" "/A001" etc..::

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

**/


inline void sreport_Creator::init_submeta()
{
    if(_creator) std::cout << "[sreport_Creator::init_submeta\n" ;
#ifdef WITH_SUBMETA
    if(_creator) std::cout << "-sreport_Creator::init_submeta.WITH_SUBMETA\n" ;

    report->submeta    = fold_valid ? fold->subfold_summary("submeta",    ASEL, BSEL) : nullptr ;
    if(_creator) std::cout << "-sreport_Creator::init_submeta :[" << ( report->submeta ? report->submeta->stats() : "-" )  << "]\n"  ;

    report->submeta_NumPhotonCollected = fold_valid ? fold->subfold_summary("submeta:NumPhotonCollected", ASEL, BSEL) : nullptr ;

    if(_creator) std::cout << "-sreport_Creator::init_submeta_NumPhotonCollected :[" << ( report->submeta_NumPhotonCollected ? report->submeta_NumPhotonCollected->stats() : "-" )  << "]\n" ;

#else
    std::cout << "-sreport_Creator::init_submeta.NOT:WITH_SUBMETA\n" ;

#endif
    if(_creator) std::cout << "]sreport_Creator::init_submeta\n" ;
}

/**
sreport_Creator::init_subcount
--------------------------------

1. create report->subcount from subfold array counts

**/

inline void sreport_Creator::init_subcount()
{
    if(_creator) std::cout << "[sreport_Creator::init_subcount\n" ;

    report->subcount   = fold_valid ? fold->subfold_summary("subcount",   ASEL, BSEL) : nullptr ;

    if(_creator) std::cout << "-sreport_Creator::init_subcount :[" << ( report->subcount ? report->subcount->stats() : "-" ) << "]\n" ;
    if(_creator) std::cout << "]sreport_Creator::init_subcount\n" ;
}


inline std::string sreport_Creator::desc() const
{
    std::stringstream ss ;
    ss << "[sreport_Creator.desc" << std::endl
       << desc_fold()
       << ( VERBOSE ? desc_fold_detail() : "" )
       << ( VERBOSE ? desc_run() : "" )
       << "]sreport_Creator.desc" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sreport_Creator::desc_fold() const
{
    std::stringstream ss ;
    ss << "[sreport_Creator.desc_fold" << std::endl
       << "fold = NPFold::LoadNoData(\"" << dirp << "\")" << std::endl
       << "fold " << ( fold ? "YES" : "NO " )  << std::endl
       << "fold_valid " << ( fold_valid ? "YES" : "NO " ) << std::endl
       << "]sreport_Creator.desc_fold" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sreport_Creator::desc_fold_detail() const
{
    std::stringstream ss ;
    ss
       << "[sreport_Creator.desc_fold_detail " << std::endl
       << ( fold ? fold->desc() : "-" ) << std::endl
       << "]sreport_Creator.desc_fold_detail " << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sreport_Creator::desc_run() const
{
    std::stringstream ss ;
    ss << "[sreport_Creator.desc_run" << std::endl
       << ( run ? run->sstr() : "-" ) << std::endl
       << ".sreport_Creator.desc_run.descMetaKVS " << std::endl
       << ( run ? run->descMetaKVS() : "-" ) << std::endl
       << "]sreport_Creator.desc_run" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}




int main(int argc, char** argv)
{
    const char* CONFIG = U::GetEnv(sreport::sreport__CONFIG,"");
    bool _main = sreport::IsConfig(CONFIG, "main") ;

    char* argv0 = argv[0] ;
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;
    if(dirp == nullptr) return 0 ;
    bool is_executable_sibling_path = U::IsExecutableSiblingPath( argv0 , dirp ) ;

    std::cout
       << "[sreport.main"
       << " CONFIG [" << ( CONFIG ? CONFIG : "-" ) << "]"
       << " argv0 " << ( argv0 ? argv0 : "-" )
       << " dirp " << ( dirp ? dirp : "-" )
       << " is_executable_sibling_path " << ( is_executable_sibling_path ? "YES" : "NO " )
       << std::endl
       ;

    if( is_executable_sibling_path == false )  // not in eg ALL3_sreport directory
    {
        U::SetEnvDefaultExecutableSiblingPath("SREPORT_FOLD", argv0, dirp );
        if(_main) std::cout << "[sreport.main : CREATING REPORT " << std::endl ;

        if(_main) std::cout << "[sreport.main : creator " << std::endl ;
        sreport_Creator creator(dirp, CONFIG);
        if(_main) std::cout << "]sreport.main : creator " << std::endl ;
        if(_main) std::cout << "[sreport.main : creator.desc " << std::endl ;
        if(_main) std::cout << creator.desc() ;
        if(_main) std::cout << "]sreport.main : creator.desc " << std::endl ;
        if(!creator.fold_valid) return 1 ;

        sreport* report = creator.report ;
        if(_main) std::cout << "[sreport.main : report.desc " << std::endl ;
        std::cout << report->desc() ;
        if(_main) std::cout << "]sreport.main : report.desc " << std::endl ;
        report->save("$SREPORT_FOLD");
        if(_main) std::cout << "]sreport.main : CREATED REPORT " << std::endl ;

        if(getenv("CHECK") != nullptr )
        {
            std::cout << "[sreport.main : CHECK LOADED REPORT " << std::endl ;
            sreport* report2 = sreport::Load("$SREPORT_FOLD") ;
            std::cout << report2->desc() ;
            std::cout << "]sreport.main : CHECK LOADED REPORT " << std::endl ;
        }

    }
    else
    {
        std::cout << "[sreport.main : LOADING REPORT " << std::endl ;
        sreport* report = sreport::Load(dirp) ;
        std::cout << report->desc() ;
        std::cout << "]sreport.main : LOADED REPORT " << std::endl ;
    }

    std::cout
       << "]sreport.main"
       << std::endl
       ;

    return 0 ;
}

