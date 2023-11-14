
#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "NP.hh"
#include "QScint.hh"

void test_check(QScint& sc)
{
    sc.check(); 
}

void test_lookup(QScint& sc)
{
    NP* dst = sc.lookup(); 
    dst->save("$TMP/QScintTest/dst.npy"); 
    sc.src->save("$TMP/QScintTest/src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = spath::Resolve("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard/icdf.npy" ); 
    NP* icdf = NP::LoadIfExists(path); 

    LOG(info) 
        << " path " << path 
        << " icdf " << ( icdf ? icdf->sstr() : "-" )
        ; 

    if(icdf == nullptr) return 0 ; 

    unsigned hd_factor = 0u ; 
    QScint sc(icdf, hd_factor);     // uploads reemission texture  

    test_lookup(sc); 
    //test_check(sc); 

    return 0 ; 
}

