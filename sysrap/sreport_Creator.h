#pragma once


#include "sreport.h"

/**
sreport_Creator
---------------

1. loads folder metadata with NPFold::LoadNoData
2. instanciates and populates sreport instance

Excercise with eg::

   sreport__CONFIG=creator cxs_min.sh report


**/

struct sreport_Creator
{
    static constexpr const char* ASEL = "a://A" ;
    static constexpr const char* BSEL = "b://B" ;

    bool VERBOSE ;
    const char* dirp ;
    const char* CONFIG ;
    bool _creator ;  // export sreport__CONFIG=creator

    NPFold*    fold ;
    bool fold_valid ;
    const NP*  run ;
    sreport*   report ;

    sreport_Creator(  const char* dirp_, const char* CONFIG);
    void init();
    void init_runprof_run_ranges_from_SProf();
    void init_evsmry_from_ranges();

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
    init_evsmry_from_ranges();

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

    report->runprof = NP::MakeMetaKVProfileArray(meta, "Index") ;  // NB: runprof seems much less featureful than ranges - BUT NEVERTHELESS KEEP RUNPROF AS runprof_meta.txt HOLDS THE FULL SProf.txt
    if(_creator) std::cout << "-sreport_Creator::init.SProf:runprof   :" << ( report->runprof ? report->runprof->sstr() : "-" ) << " {runprof: grep Index SProf.txt, with full SProf.txt in runprof_meta.txt}" << std::endl ;

    std::stringstream ss ;
    report->ranges = NP::MakeMetaKVS_ranges2( meta, sreport::RANGES, &ss ) ;
    std::string str = ss.str();
    if(_creator)
    {
         std::cout
             << "--sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2[\n"
             << str
             << "--sreport_Creator::init_runprof_run_ranges_from_SProf.ranges2]\n"
             ;
    }

    if(_creator) std::cout << "-sreport_Creator::init_SProf.ranges2   :" << ( report->ranges ?  report->ranges->sstr() : "-" ) << " {ranges provides detailed timing info extracted from SProf.txt}" <<  std::endl ;


    report->run     = run ? run->copy() : nullptr ;
    if(_creator) std::cout << "-sreport_Creator::init_SProf.run       :" << ( report->run ? report->run->sstr() : "-" ) << " {run is dummy array just for holding metadata} " << std::endl ;

    if(_creator) std::cout << "]sreport_Creator::init_runprof_run_ranges_from_SProf\n" ;
}



inline void sreport_Creator::init_evsmry_from_ranges()
{
    if(_creator) std::cout << "[sreport_Creator::init_evsmry_from_ranges\n" ;

    std::stringstream ss ;
    report->evsmry = NP::Summarize_ranges_to_evsmry( report->ranges, &ss );
    std::string str = ss.str();
    if(_creator)
    {
         std::cout
             << "-sreport_Creator::init_evsmry_from_ranges.Summarize_ranges_to_evsmry[\n"
             << str
             << "-sreport_Creator::init_evsmry_from_ranges.Summarize_ranges_to_evsmry]\n"
             ;
    }
    if(_creator) std::cout << "]sreport_Creator::init_evsmry_from_ranges\n" ;
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


