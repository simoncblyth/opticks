#include "Opticks.hh"
#include "NP.hh"
#include "QCerenkov.hh"
#include "scuda.h"
#include "OPTICKS_LOG.hh"


void test_check(QCerenkov& ck)
{
    ck.check(); 
}

void test_lookup(QCerenkov& sc)
{
    NP* dst = sc.lookup(); 
    const char* fold = "$TMP/QCerenkovTest" ; 
    LOG(info) << " save to " << fold ; 
    dst->save(fold, "dst.npy"); 
    sc.src->save(fold, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    QCerenkov ck ;  

    //test_lookup(ck); 
    //test_check(ck); 

    return 0 ; 
}

