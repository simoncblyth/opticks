#include "Opticks.hh"
#include "NPY.hpp"
#include "GScintillatorLib.hh"
#include "QScint.hh"
#include "scuda.h"
#include "OPTICKS_LOG.hh"


void test_check(QScint& sc)
{
    sc.check(); 
}

void test_lookup(QScint& sc)
{
    NPY<float>* dst = sc.lookup(); 
    const char* fold = "$TMP/QScintTest" ; 
    LOG(info) << " save to " << fold ; 
    dst->save(fold, "dst.npy"); 
    sc.src->save(fold, "src.npy") ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    QScint sc(slib);     // uploads reemission texture  

    test_lookup(sc); 
    //test_check(sc); 

    return 0 ; 
}

