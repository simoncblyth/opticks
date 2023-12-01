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
| a   | "//p" eg: //p001 //p002         | Opticks/QSim SEvt       |
+-----+---------------------------------+-------------------------+
| b   | "//n" eg: //n001 //n002         | Geant4/U4Recorder SEvt  |
+-----+---------------------------------+-------------------------+

The tables are presented with row and column labels and the 
summary NPFold is saved to DIR_sreport sibling to invoking DIR 
which needs to contain SEvt/NPFold folders corresponding to the path prefix.  
The use of NPFold::LoadNoData means that only SEvt NPFold/NP 
metadata is loaded. Excluding the array data makes the load 
very fast and able to handle large numbers of persisted SEvt NPFold.

Usage::

    epsilon:~ blyth$ cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0
    epsilon:ALL0 blyth$ sreport 
    epsilon:ALL0 blyth$ ls -alst ../ALL0_sreport 
    total 8
    8 -rw-r--r--  1 blyth  staff    4 Nov 26 14:12 NPFold_index.txt
    0 drwxr-xr-x  9 blyth  staff  288 Nov 26 13:01 b
    0 drwxr-xr-x  5 blyth  staff  160 Nov 26 13:01 .
    0 drwxr-xr-x  9 blyth  staff  288 Nov 26 13:01 a
    0 drwxr-xr-x  5 blyth  staff  160 Nov 26 13:01 ..
    epsilon:ALL0 blyth$ 



TODO: restructure to allow loading summary report 
and giving the textual presentation by refactor into 
create/save/load/desc methods.

Can detect whether to create or load based on 
the name of the invoking or argument directory. 

**/

#include "NPFold.h"

struct sreport
{
    static constexpr const char* ASEL = "a://p" ; 
    static constexpr const char* BSEL = "b://n" ; 

    const char* dirp ; 
    NPFold* fold ; 
    bool fold_valid ; 

    bool VERBOSE ;
    const NP* run ;
    const NP* runprof ;
    NPFold* substamp ;   
    NPFold* subprofile ;   
    NPFold* smry ; 

    sreport(const char* dirp);  
    void init(); 

    std::string desc() const ;
    std::string desc_fold() const ;
    std::string desc_run() const ;
    std::string desc_runprof() const ;
    std::string desc_substamp() const ;
    std::string desc_subprofile() const ;

    void save(const char* dir); 
};

inline sreport::sreport( const char* dirp_ )
    :
    dirp(dirp_),
    fold(NPFold::LoadNoData(dirp)),
    fold_valid( fold && !fold->is_empty() ),
    VERBOSE(getenv("sreport__VERBOSE") != nullptr),
    run(fold ? fold->get("run") : nullptr),
    runprof( run ? run->makeMetaKVProfileArray("Index") : nullptr),
    substamp(fold_valid ? fold->subfold_summary("substamp", ASEL, BSEL) : nullptr),
    subprofile(fold_valid ? fold->subfold_summary("subprofile", ASEL, BSEL) : nullptr),
    smry(new NPFold)
{
    smry->add("runprof", runprof ) ; 
    smry->add_subfold("substamp", substamp ) ; 
    smry->add_subfold("subprofile", subprofile ) ; 
}


inline std::string sreport::desc() const
{
    std::stringstream ss ; 
    ss << "[sreport.desc" << std::endl 
       << desc_fold()
       << desc_run() 
       << desc_runprof() 
       << desc_substamp()
       << "]sreport.desc" << std::endl 
       ; 
    std::string str = ss.str() ;
    return str ;  
}

inline std::string sreport::desc_fold() const
{
    std::stringstream ss ; 
    ss << "[sreport.desc_fold" << std::endl 
       << "fold = NPFold::LoadNoData(\"" << dirp << "\")" << std::endl
       << "fold " << ( fold ? "YES" : "NO " )  << std::endl
       << "fold_valid " << ( fold_valid ? "YES" : "NO " ) << std::endl
       ;

    if(VERBOSE) ss
       << "[sreport.desc_fold.VERBOSE " << std::endl
       << ( fold ? fold->desc() : "-" ) << std::endl
       << "]sreport.desc_fold.VERBOSE " << std::endl
       ; 

    ss << "]sreport.desc_fold" << std::endl 
       ; 

    std::string str = ss.str() ;
    return str ;  
}

inline std::string sreport::desc_run() const
{
    std::stringstream ss ; 
    ss << "[sreport.desc_run" << std::endl 
       << ( run ? run->sstr() : "-" ) << std::endl 
       << ".sreport.desc_run.descMetaKVS " << std::endl 
       << ( run ? run->descMetaKVS() : "-" ) << std::endl
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


inline void sreport::save(const char* dir)
{
    smry->save_verbose(dir); 
}

int main(int argc, char** argv)
{
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;   
    if(dirp == nullptr) return 0 ; 
    U::SetEnvDefaultExecutableSiblingPath("FOLD", argv[0], dirp );

    sreport rep(dirp) ; 
    std::cout << rep.desc() ; 
    if(!rep.fold_valid) return 1 ; 
    rep.save("$FOLD"); 
    return 0 ; 
}

