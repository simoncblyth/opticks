/**
QBndTest.cc
------------

TOOD: consolidate QBnd and QOptical 



**/

#include "scuda.h"
#include "SStr.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SPath.hh"
#include "NP.hh"
#include "SOpticksResource.hh"

#include "QBnd.hh"
#include "QTex.hh"
#include "OPTICKS_LOG.hh"


/**
test_lookup_technical
----------------------

Technical test doing lookups over the entire texture.
TODO: a test more like actual usage.

**/

void test_lookup_technical(QBnd& qb)
{
    NP* lookup = qb.lookup(); 
    const char* dir = SPath::Resolve("$TMP/QBndTest", DIRPATH) ; 
    LOG(info) << " save to " << dir  ; 
    lookup->save(dir, "dst.npy"); 
    qb.src->save(dir, "src.npy") ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cfbase = SOpticksResource::CFBase() ; 
    LOG(info) << " cfbase " << cfbase ; 
    NP* bnd = NP::Load(cfbase, "CSGFoundry/SSim/bnd.npy"); 

    QBnd qb(bnd) ; 

/*
    test_lookup_technical(qb); 
*/
    return 0 ; 
}
