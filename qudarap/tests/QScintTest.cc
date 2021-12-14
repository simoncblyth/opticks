#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

#include "Opticks.hh"

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
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve("$TMP/QScintTest", create_dirs) ; 
    LOG(info) << " save to " << fold ; 
    dst->save(fold, "dst.npy"); 
    sc.src->save(fold, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

#ifdef OLD_WAY
    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();
    NP* icdf = slib->getBuf(); 
#else
    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
    NP* icdf = NP::Load(cfbase, "CSGFoundry", "icdf.npy"); // HMM: this needs a more destinctive name/location  
#endif
    //icdf->dump(); 

    unsigned hd_factor = 0u ; 

    QScint sc(icdf, hd_factor);     // uploads reemission texture  

    test_lookup(sc); 
    //test_check(sc); 

    return 0 ; 
}

