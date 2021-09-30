#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#ifdef OLD_WAY
#include "Opticks.hh"
#include "GScintillatorLib.hh"
#endif

#include "QScint.hh"
#include "scuda.h"
#include "OPTICKS_LOG.hh"


void test_check(QScint& sc)
{
    sc.check(); 
}

void test_lookup(QScint& sc)
{
    NP* dst = sc.lookup(); 
    const char* fold = SPath::Resolve("$TMP/QScintTest") ; 
    LOG(info) << " save to " << fold ; 
    dst->save(fold, "dst.npy"); 
    sc.src->save(fold, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef OLD_WAY
    Opticks ok(argc, argv); 
    ok.configure(); 
    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();
    NP* icdf = slib->getBuf(); 
#else
    const char* cfbase = SPath::Resolve(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ));
    NP* icdf = NP::Load(cfbase, "CSGFoundry", "icdf.npy"); // HMM: this needs a more destinctive name/location  
    //icdf->dump(); 
#endif

    unsigned hd_factor = 0u ; 

    QScint sc(icdf, hd_factor);     // uploads reemission texture  

    test_lookup(sc); 
    //test_check(sc); 

    return 0 ; 
}

