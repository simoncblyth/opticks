#include "NP.hh"
#include "Opticks.hh"
#include "QProp.hh"
#include "OPTICKS_LOG.hh"

const char* FOLD = "/tmp/QPropTest" ; 

/**
test_lookup
-------------

nx lookups in x0->x1 inclusive for each property yielding nx*qp.ni values.

**/

void test_lookup(const QProp& qp, float x0, float x1, unsigned nx)
{
    NP* x = NP::Linspace<float>( x0, x1, nx ); 
    NP* y = NP::Make<float>(qp.ni, nx ); 

    qp.lookup(y->values<float>(), x->cvalues<float>(), qp.ni, nx );

    qp.a->save(FOLD, "prop.npy"); 
    x->save(FOLD, "domain.npy"); 
    y->save(FOLD, "lookup.npy"); 



    LOG(info) << "save to " << FOLD ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    QProp qp ; 
    test_lookup(qp, 0.f, 16.f, 1601u ); 

    return 0 ; 
}

