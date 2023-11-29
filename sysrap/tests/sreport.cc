/**
sreport.cc : Summarize + Present SEvt/NPFold metadata time stamps 
=============================================================================

* formerly sstampfold_report using sstampfold.h but that functionality 
  moved into NPFold.h NPX.h 

::
 
    ~/opticks/sysrap/tests/sreport.sh 
    JOB=N2 ~/opticks/sysrap/tests/sreport.sh runo


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
summary NPFold is saved to ./sreport relative to 
the invoking directory which needs to contain SEvt/NPFold folders 
corresponding to the path prefix.  

The use of NPFold::LoadNoData means that only SEvt NPFold/NP 
metadata is loaded. Excluding the array data makes the load 
very fast and able to handle large numbers of persisted SEvt NPFold. 

Usage::

    epsilon:~ blyth$ cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0

    epsilon:ALL0 blyth$ find . -name NPFold_meta.txt | head -5
    ./p009/NPFold_meta.txt
    ./p007/NPFold_meta.txt
    ./p001/NPFold_meta.txt
    ./p006/NPFold_meta.txt
    ./p008/NPFold_meta.txt

    epsilon:ALL0 blyth$ sreport 
    ...
    
    epsilon:ALL0 blyth$ ls -alst ../sreport/   ##  ls output NPFold directory 
    total 8
    8 -rw-r--r--  1 blyth  staff    4 Nov 26 14:12 NPFold_index.txt
    0 drwxr-xr-x  9 blyth  staff  288 Nov 26 13:01 b
    0 drwxr-xr-x  5 blyth  staff  160 Nov 26 13:01 .
    0 drwxr-xr-x  9 blyth  staff  288 Nov 26 13:01 a
    0 drwxr-xr-x  5 blyth  staff  160 Nov 26 13:01 ..
    epsilon:ALL0 blyth$ 



Actions
---------

1. NPFold::LoadNoData metadata of a single run with run metdata and 
   multiple SEvt folders often with both //p and //n prefix subfold

2. 




**/

#include <cstdlib>

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
    NPFold* substamp ;   
    NPFold* subprofile ;   
    NPFold* smry ; 

    sreport(const char* dirp);  
    void init(); 

    std::string desc() const ;
    std::string desc_fold() const ;
    std::string desc_run() const ;
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
    substamp(fold_valid ? fold->subfold_summary("substamp", ASEL, BSEL) : nullptr),
    subprofile(fold_valid ? fold->subfold_summary("subprofile", ASEL, BSEL) : nullptr),
    smry(new NPFold)
{
    smry->add_subfold("substamp", substamp ) ; 
    smry->add_subfold("subprofile", subprofile ) ; 
}


inline std::string sreport::desc() const
{
    std::stringstream ss ; 
    ss << "[sreport.desc" << std::endl 
       << desc_fold()
       << desc_run() 
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

