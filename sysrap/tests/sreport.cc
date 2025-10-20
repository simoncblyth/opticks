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

    bool    VERBOSE ;

    NP*       run ;   // dummy array that exists just to hold metadata
    NP*       runprof ;
    NP*       ranges ;

    NPFold*   substamp ;
    NPFold*   subprofile ;
    NPFold*   submeta ;
    NPFold*   submeta_NumPhotonCollected ;
    NPFold*   subcount ;

    sreport();

    NPFold* serialize() const ;
    void    import( const NPFold* fold );
    void save(const char* dir) const ;
    static sreport* Load(const char* dir) ;

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
    VERBOSE(getenv("sreport__VERBOSE") != nullptr),
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

inline std::string sreport::desc() const
{
    std::stringstream ss ;
    ss << "[sreport.desc" << std::endl
       << desc_run()
       << desc_runprof()
       << desc_ranges()
       << desc_substamp()
       << desc_submeta()
       << desc_subcount()
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
    ss << "[sreport.desc_runprof" << std::endl
       << ( runprof ? runprof->sstr() : "-" ) << std::endl
       << ".sreport.desc_runprof.descTable " << std::endl
       << ( runprof ? runprof->descTable<int64_t>(17) : "-" ) << std::endl
       << "]sreport.desc_runprof" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string sreport::desc_ranges() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_ranges"
       << " ranges : " << ( ranges ? ranges->sstr() : "-" ) << std::endl
       << ".sreport.desc_ranges.descTable "
       << " ( ta,tb : timestamps expressed as seconds from first timestamp, ab: (tb-ta) )"
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


inline std::string sreport::desc_submeta() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_submeta" << std::endl
       << ( submeta ? submeta->desc() : "-" )
       << "]sreport.desc_submeta" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}
inline std::string sreport::desc_subcount() const
{
    std::stringstream ss ;
    ss << "[sreport.desc_subcount" << std::endl
       << ( subcount ? subcount->desc() : "-" )
       << "]sreport.desc_subcount" << std::endl
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
    NPFold*    fold ;
    bool fold_valid ;
    const NP*  run ;
    sreport*   report ;

    sreport_Creator(  const char* dirp_ );
    void init();
    void init_SProf();
    void init_substamp();
    void init_subprofile();
    void init_submeta();
    void init_subcount();

    std::string desc() const ;
    std::string desc_fold() const ;
    std::string desc_fold_detail() const ;
    std::string desc_run() const ;
};

inline sreport_Creator::sreport_Creator( const char* dirp_ )
    :
    VERBOSE(getenv("sreport_Creator__VERBOSE") != nullptr),
    dirp(dirp_ ? strdup(dirp_) : nullptr),
    fold(NPFold::LoadNoData(dirp)),
    fold_valid(NPFold::IsValid(fold)),
    run(fold_valid ? fold->get("run") : nullptr),
    report(new sreport)
{
    std::cout
        << "[sreport_Creator::sreport_Creator"
        << " fold_valid " << ( fold_valid ? "YES" : "NO " )
        << " run " << ( run ? "YES" : "NO " )
        << "\n"
        ;
    init();
    std::cout << "]sreport_Creator::sreport_Creator" << std::endl ;
}


/**
sreport_Creator::init
-----------------------

1. construct SProf derived metadata arrays
2. construct subfold derived arrays and fold

**/


inline void sreport_Creator::init()
{
    std::cout << "[sreport_Creator::init\n" ;

    init_SProf();

    init_substamp();
    init_subprofile();
    init_submeta();
    init_subcount();

    std::cout << "]sreport_Creator::init\n" ;
}

/**
sreport_Creator::init_SProf
----------------------------

The SProf has the advantage of almost always being available, as the SProf.txt is small
unlike the full arrays.

1. read SProf.txt into meta string
2. create report->runprof array from the "Index" lines, shaped (2,3) for the below example
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

**/

inline void sreport_Creator::init_SProf()
{
    std::cout << "[sreport_Creator::init_SProf\n" ;

    std::string meta = U::ReadString2_("SProf.txt");

    report->runprof = NP::MakeMetaKVProfileArray(meta, "Index") ;
    std::cout << "-sreport_Creator::init.SProf:runprof   :" << ( report->runprof ? report->runprof->sstr() : "-" ) << std::endl ;
    // report->runprof, should now be report->prof

    report->run     = run ? run->copy() : nullptr ;
    std::cout << "-sreport_Creator::init_SProf.run       :" << ( report->run ? report->run->sstr() : "-" ) << std::endl ;

    report->ranges = run ? NP::MakeMetaKVS_ranges2( meta, sreport::RANGES ) : nullptr ;
    std::cout << "-sreport_Creator::init_SProf.ranges2   :" << ( report->ranges ?  report->ranges->sstr() : "-" ) <<  std::endl ;

    std::cout << "]sreport_Creator::init_SProf\n" ;
}

/**
sreport_Creator::init_substamp
-------------------------------

1. create report->substamp from stamps found in NPFold_meta.txt of the specified subfolders

**/


inline void sreport_Creator::init_substamp()
{
    std::cout << "[sreport_Creator::init_substamp\n" ;

    std::cout << "-sreport_Creator::init_substamp fold_valid " << ( fold_valid ? "Y" : "N" ) << std::endl ;

    report->substamp   = fold_valid ? fold->subfold_summary("substamp",   ASEL, BSEL) : nullptr ;
    std::cout << "-sreport_Creator::init_substamp ((NPFold)report.substamp).stats [" << ( report->substamp ? report->substamp->stats() : "-" ) << "]\n" ;

    std::cout << "]sreport_Creator::init_substamp\n" ;
}

/**
sreport_Creator::init_subprofile
----------------------------------

1. create report->subprofile from profile metadata from the subfold

**/

inline void sreport_Creator::init_subprofile()
{
    std::cout << "[sreport_Creator::init_subprofile\n" ;
    report->subprofile = fold_valid ? fold->subfold_summary("subprofile", ASEL, BSEL) : nullptr ;
    std::cout << "-sreport_Creator::init_subprofile :[" << ( report->subprofile ? report->subprofile->stats() : "-" )  << "]\n" ;
    std::cout << "]sreport_Creator::init_subprofile\n" ;
}

/**
sreport_Creator::init_submeta
-------------------------------

1. create report->submeta from subfold metadata
2. report->submeta_NumPhotonCollected from subfold metadata

**/


inline void sreport_Creator::init_submeta()
{
    std::cout << "[sreport_Creator::init_submeta\n" ;
#ifdef WITH_SUBMETA
    std::cout << "-sreport_Creator::init_submeta.WITH_SUBMETA\n" ;

    report->submeta    = fold_valid ? fold->subfold_summary("submeta",    ASEL, BSEL) : nullptr ;
    std::cout << "-sreport_Creator::init_submeta :[" << ( report->submeta ? report->submeta->stats() : "-" )  << "]\n"  ;

    report->submeta_NumPhotonCollected = fold_valid ? fold->subfold_summary("submeta:NumPhotonCollected", ASEL, BSEL) : nullptr ;
    std::cout << "-sreport_Creator::init_submeta_NumPhotonCollected :[" << ( report->submeta_NumPhotonCollected ? report->submeta_NumPhotonCollected->stats() : "-" )  << "]\n" ;

#else
    std::cout << "-sreport_Creator::init_submeta.NOT:WITH_SUBMETA\n" ;

#endif
    std::cout << "]sreport_Creator::init_submeta\n" ;
}

/**
sreport_Creator::init_subcount
--------------------------------

1. create report->subcount from subfold array counts

**/

inline void sreport_Creator::init_subcount()
{
    std::cout << "[sreport_Creator::init_subcount\n" ;
    report->subcount   = fold_valid ? fold->subfold_summary("subcount",   ASEL, BSEL) : nullptr ;
    std::cout << "-sreport_Creator::init_subcount :[" << ( report->subcount ? report->subcount->stats() : "-" ) << "]\n" ;
    std::cout << "]sreport_Creator::init_subcount\n" ;
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
    char* argv0 = argv[0] ;
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;
    if(dirp == nullptr) return 0 ;
    bool is_executable_sibling_path = U::IsExecutableSiblingPath( argv0 , dirp ) ;

    std::cout
       << "[sreport.main"
       << "  argv0 " << ( argv0 ? argv0 : "-" )
       << " dirp " << ( dirp ? dirp : "-" )
       << " is_executable_sibling_path " << ( is_executable_sibling_path ? "YES" : "NO " )
       << std::endl
       ;

    if( is_executable_sibling_path == false )  // not in eg ALL3_sreport directory
    {
        U::SetEnvDefaultExecutableSiblingPath("SREPORT_FOLD", argv0, dirp );
        std::cout << "[sreport.main : CREATING REPORT " << std::endl ;

        std::cout << "[sreport.main : creator " << std::endl ;
        sreport_Creator creator(dirp);
        std::cout << "]sreport.main : creator " << std::endl ;
        std::cout << "[sreport.main : creator.desc " << std::endl ;
        std::cout << creator.desc() ;
        std::cout << "]sreport.main : creator.desc " << std::endl ;
        if(!creator.fold_valid) return 1 ;

        sreport* report = creator.report ;
        std::cout << "[sreport.main : report.desc " << std::endl ;
        std::cout << report->desc() ;
        std::cout << "]sreport.main : report.desc " << std::endl ;
        report->save("$SREPORT_FOLD");
        std::cout << "]sreport.main : CREATED REPORT " << std::endl ;

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

