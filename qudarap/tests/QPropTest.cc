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

void test_lookup(QProp& qp, float x0, float x1, unsigned nx)
{
    NP* x = NP::Linspace<float>( x0, x1, nx ); 
    NP* y = NP::Make<float>(qp.ni, nx ); 

    qp.lookup(y->values<float>(), x->cvalues<float>(), qp.ni, nx );

    x->save(FOLD, "domain.npy"); 
    y->save(FOLD, "lookup.npy"); 

    LOG(info) << "save to " << FOLD ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* path = "/tmp/np/test_compound_np_interp.npy" ; 
    NP* a = NP::Load(path); 
    a->save(FOLD, "prop.npy"); 

    LOG(info) << " a " << ( a ? a->desc() : "-" ); 
    if( a == nullptr ) return 0 ; 

    QProp qp(a) ; 
    test_lookup(qp, 0.f, 10.f, 11u ); 

    return 0 ; 
}

