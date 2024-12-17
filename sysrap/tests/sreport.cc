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
        A%0.3d_QSim__simulate_HEAD:A%0.3d_QSim__simulate_PREL         ## upload_genstep
        A%0.3d_QSim__simulate_PREL:A%0.3d_QSim__simulate_POST         ## simulate
        A%0.3d_QSim__simulate_POST:A%0.3d_QSim__simulate_TAIL         ## download 
       )" ; 

    bool    VERBOSE ;

    NP*       run ; 
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
    ss << "[sreport.desc_run (run is dummy small array used as somewhere to hang metadata) " << std::endl 
       << ( run ? run->sstr() : "-" ) << std::endl 
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
    ss << "[sreport.desc_ranges" << std::endl 
       << ( ranges ? ranges->sstr() : "-" ) << std::endl 
       << ".sreport.desc_ranges.descTable " << std::endl 
       << ( ranges ? ranges->descTable<int64_t>(17) : "-" ) << std::endl
       << "]sreport.desc_ranges" << std::endl 
       ; 
    std::string str = ss.str() ;
    return str ;  
}




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

inline void sreport_Creator::init() 
{
    std::cout << "[sreport_Creator::init" << std::endl ; 

    report->runprof = run ? run->makeMetaKVProfileArray("Index") : nullptr ; 
    std::cout << "-sreport_Creator::init.1:runprof   :" << ( report->runprof ? report->runprof->sstr() : "-" ) << std::endl ; 

    report->run     = run ? run->copy() : nullptr ; 
    std::cout << "-sreport_Creator::init.2.run       :" << ( report->run ? report->run->sstr() : "-" ) << std::endl ; 

    report->ranges = run ? NP::MakeMetaKVS_ranges2( run->meta, sreport::RANGES ) : nullptr ; 
    std::cout << "-sreport_Creator::init.3.ranges2   :" << ( report->ranges ?  report->ranges->sstr() : "-" ) <<  std::endl ; 


    std::cout << "-sreport_Creator::init.4 fold_valid " << ( fold_valid ? "Y" : "N" ) << std::endl ; 

    report->substamp   = fold_valid ? fold->subfold_summary("substamp",   ASEL, BSEL) : nullptr ; 
    std::cout << "-sreport_Creator::init.4.substamp   :[" << ( report->substamp ? report->substamp->stats() : "-" ) << "]\n" ; 

    report->subprofile = fold_valid ? fold->subfold_summary("subprofile", ASEL, BSEL) : nullptr ; 
    std::cout << "-sreport_Creator::init.5.subprofile :[" << ( report->subprofile ? report->subprofile->stats() : "-" )  << "]\n" ; 


#ifdef WITH_SUBMETA
    std::cout << "-sreport_Creator::init.6.WITH_SUBMETA" << std::endl ; 

    report->submeta    = fold_valid ? fold->subfold_summary("submeta",    ASEL, BSEL) : nullptr ; 
    std::cout << "-sreport_Creator::init.7.submeta :[" << ( report->submeta ? report->submeta->stats() : "-" )  << "]\n"  ; 

    report->submeta_NumPhotonCollected = fold_valid ? fold->subfold_summary("submeta:NumPhotonCollected", ASEL, BSEL) : nullptr ; 
    std::cout << "-sreport_Creator::init.8.submeta_NumPhotonCollected :[" << ( report->submeta_NumPhotonCollected ? report->submeta_NumPhotonCollected->stats() : "-" )  << "]\n" ; 

#else
    std::cout << "-sreport_Creator::init.6.NOT:WITH_SUBMETA" << std::endl ; 

#endif

    report->subcount   = fold_valid ? fold->subfold_summary("subcount",   ASEL, BSEL) : nullptr ; 
    std::cout << "-sreport_Creator::init.9.subcount :[" << ( report->subcount ? report->subcount->stats() : "-" ) << "]\n" ; 

    std::cout << "]sreport_Creator::init" << std::endl ; 
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

