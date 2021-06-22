
#include "scuda.h"
#include "NPY.hpp"
#include "Opticks.hh"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "QBnd.hh"
#include "OPTICKS_LOG.hh"

void test_lookup(QBnd& qb)
{
    NPY<float>* lookup = qb.lookup(); 
    const char* dir = "$TMP/QBndTest" ; 
    LOG(info) << " save to " << dir  ; 
    lookup->save(dir, "dst.npy"); 
    qb.src->save(dir, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    GGeo* gg = GGeo::Load(&ok); 
    GBndLib* blib = gg->getBndLib(); 
    blib->createDynamicBuffers();  // hmm perhaps this is done already on loading now ?

    QBnd qb(blib) ; 

    test_lookup(qb); 

    return 0 ; 
}
