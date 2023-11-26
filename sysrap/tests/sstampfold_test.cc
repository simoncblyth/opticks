/**
sstampfold_test.cc : Summarize + Present SEvt/NPFold metadata time stamps 
===========================================================================

Summarizes SEvt/NPFold metadata time stamps into substamp arrays 
grouped by NPFold path prefix. The summary NPFold is presented textually 
and saved to allow plotting from python. 

TODO: bring over the plotting machinery from ~/np/tests/NPFold_stamps_test.py 

+-----+---------------------------------+-------------------------+
| key | SEvt/NPFold path prefix         |  SEvt type              |
+=====+=================================+=========================+
| a   | "//p" eg: //p001 //p002         | Opticks/QSim SEvt       |
+-----+---------------------------------+-------------------------+
| b   | "//n" eg: //n001 //n002         | Geant4/U4Recorder SEvt  |
+-----+---------------------------------+-------------------------+

The tables are presented with row and column labels and the 
summary NPFold is saved to ./sstampfold_test relative to 
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

    epsilon:ALL0 blyth$ sstampfold_test 
    ...
    
    epsilon:ALL0 blyth$ ls -alst ../sstampfold_test/   ##  ls output NPFold directory 
    total 8
    8 -rw-r--r--  1 blyth  staff    4 Nov 26 14:12 NPFold_index.txt
    0 drwxr-xr-x  9 blyth  staff  288 Nov 26 13:01 b
    0 drwxr-xr-x  5 blyth  staff  160 Nov 26 13:01 .
    0 drwxr-xr-x  9 blyth  staff  288 Nov 26 13:01 a
    0 drwxr-xr-x  5 blyth  staff  160 Nov 26 13:01 ..
    epsilon:ALL0 blyth$ 


**/

#include <cstdlib>
#include "sstampfold.h"


int main(int argc, char** argv)
{
    const char* dirp = argc > 1 ? argv[1] : U::PWD() ;   
    if(dirp == nullptr) return 0 ; 
    U::SetEnvDefaultExecutableSiblingPath("FOLD", argv[0], dirp );

    bool VERBOSE = getenv("VERBOSE") != nullptr ; 
 
    NPFold* f = NPFold::LoadNoData(dirp); 

    std::cout 
        << "sstampfold_test"
        << std::endl
        << "NPFold::LoadNoData(\"" << dirp << "\")" 
        << std::endl
        ;

    if(VERBOSE || true) std::cout 
        << "[sstampfold_test.VERBOSE "
        << std::endl
        << f->desc()
        << std::endl
        << "]sstampfold_test.VERBOSE "
        << std::endl
        ; 

    NPFold* smry = f->subfold_summary(sstampfold::STAMP_KEY, "a://p", "b://n"); 
    std::cout << smry->desc() << std::endl ;

    const NPFold* a = smry->find_subfold("a"); 
    const NPFold* b = smry->find_subfold("b"); 

    sstampfold ast(a, "ast"); 
    sstampfold bst(b, "bst"); 

    std::cout << ast.desc() ; 
    std::cout << bst.desc() ; 

    smry->save_verbose("$FOLD"); 

    return 0 ; 
}

