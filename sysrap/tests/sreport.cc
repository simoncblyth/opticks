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


int main(int argc, char** argv)
{
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;   
    if(dirp == nullptr) return 0 ; 
    U::SetEnvDefaultExecutableSiblingPath("FOLD", argv[0], dirp );

    bool VERBOSE = getenv("VERBOSE") != nullptr ; 

    NPFold* f = NPFold::LoadNoData(dirp); 
    bool f_empty = f && f->is_empty()  ; 

    std::cout 
        << "sreport"
        << std::endl
        << "NPFold::LoadNoData(\"" << dirp << "\")" 
        << std::endl
        << " f " << ( f ? "YES" : "NO " ) 
        << std::endl
        << " f_empty " << ( f_empty ? "YES" : "NO " )
        << std::endl
        ;

    if(f_empty) std::cerr 
        << "sreport : ABORT : f_empty : EMPTY DIR ? " 
        << std::endl 
        << ( dirp ? dirp : "-" )
        << std::endl 
        ;
    if(f_empty) return 1 ; 

    const NP* run = f->get("run"); 
    if(run) std::cout 
        << "[sreport.run " 
        << run->sstr() 
        << std::endl 
        << " sreport.run.descMetaKVS "
        << std::endl 
        << run->descMetaKVS()
        << "]sreport.run " 
        << std::endl 
        ;   

    if(VERBOSE) std::cout 
        << "[sreport.VERBOSE "
        << std::endl
        << f->desc()
        << std::endl
        << "]sreport.VERBOSE "
        << std::endl
        ; 

    NPFold* smry = f->subfold_summary("substamp", "a://p", "b://n"); 

    if(VERBOSE) std::cout 
        << "[sreport.smry.desc.VERBOSE" << std::endl 
        << smry->desc() 
        << "]sreport.smry.desc.VERBOSE" << std::endl 
        << std::endl
        ;

    std::cout << smry->compare_subarrays_report<double, int64_t>( "delta_substamp", "a", "b" ) ; 


    smry->save_verbose("$FOLD"); 

    return 0 ; 
}

