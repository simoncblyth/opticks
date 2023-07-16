#include "NP.hh"

#include "SProp.hh"
#include "QProp.hh"
#include "OPTICKS_LOG.hh"

const char* FOLD = "/tmp/QPropTest" ; 

/**
test_lookup
-------------

nx lookups in x0->x1 inclusive for each property yielding nx*qp.ni values.

1. create *x* domain array of shape (nx,) with values in range x0 to x1 
2. create *y* lookup array of shape (qp.ni, nx ) 
3. invoke QProp::lookup collecting *y* lookup values from kernel call 
4. save prop, domain and lookup into fold/reldir

**/

template <typename T>
void test_lookup(const QProp<T>& qp, T x0, T x1, T nx, const char* fold, const char* reldir)
{
    NP* x = NP::Linspace<T>( x0, x1, nx ); 
    NP* y = NP::Make<T>(qp.ni, nx ); 
    const NP* a = qp.a ; 

    qp.lookup(y->values<T>(), x->cvalues<T>(), qp.ni, nx );

    a->save(fold, reldir, "prop.npy"); 
    x->save(fold, reldir, "domain.npy"); 
    y->save(fold, reldir, "lookup.npy"); 

    LOG(info) << "save to " << fold << "/" << reldir  ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const char* BASE = "$CFBaseFromGEOM" ; 
    const char* BASE = "$HOME/.opticks/GEOM/$GEOM" ; 
    const char* RELP = "GGeo/GScintillatorLib/LS_ori/RINDEX.npy" ; 
    const NP* propcom = SProp::MockupCombination( BASE, RELP);

    LOG(info) << " propcom " << ( propcom ? propcom->sstr() : "-" ) ; 

    unsigned nx = 1601u ; 
    //unsigned nx = 161u ; 

    QProp<float> qpf(propcom) ; 
    test_lookup<float>(qpf, 0.f, 16.f, nx , FOLD, "float" );

    //QProp<double> qpd(propcom) ; 
    //test_lookup<double>(qpd, 0., 16., nx , FOLD, "double" );

    return 0 ; 
}

