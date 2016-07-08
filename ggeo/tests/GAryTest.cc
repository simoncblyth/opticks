#include <cassert>
#include "GDomain.hh"
#include "GAry.hh"

#include "PLOG.hh"
#include "NPY_LOG.hh"

typedef GAry<float> A ;


void test_cie()
{
   GDomain<float>* nmdom = new GDomain<float>(380.f, 780.f, 20.f );
   nmdom->Summary("nmdom");

   A* nm = A::from_domain(nmdom);
   nm->Summary("nm from_domain");

   A* X = A::cie_X( nm ); 
   A* Y = A::cie_Y( nm ); 
   A* Z = A::cie_Z( nm ); 

   X->Summary("cie_X");
   Y->Summary("cie_Y");
   Z->Summary("cie_Z");

}


void test_planck()
{
   GDomain<float>* nmdom = new GDomain<float>(60.f, 810.f, 20.f );
   nmdom->Summary("nmdom");

   A* nm = A::from_domain(nmdom);
   nm->Summary("nm from_domain");

   A* bb = A::planck_spectral_radiance( nm, 6500.f ); 
   bb->Summary("bb planck_spectral_radiance");

   nm->save("$TMP/GAry_test_planck_nm.npy");
   bb->save("$TMP/GAry_test_planck_bb.npy");
}


void test_misc()
{
    A* a = A::from_constant(10, 1.f) ;
    a->Summary("a");

    A* c = a->cumsum();
    c->Summary("a->cumsum()");

    A* u = A::urandom();
    u->Summary("u");
    u->save("$TMP/urandom.npy");
}


void test_sliced()
{
    A* a = A::linspace(11, 0, 1);
    a->Summary("A::linspace(11, 0, 1)");
    assert(a->getLength() == 11 );

    A* b = a->sliced(0, 11);
    b->Summary("a->sliced(0,11)");
    assert(b->getLength() == 11 );

    A* c = a->sliced(0, -1);
    c->Summary("a->sliced(0,-1)");
    assert(c->getLength() == 10 );

    A* d = a->sliced(0, 10);
    d->Summary("a->sliced(0,10)");
    assert(d->getLength() == 10 );
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG_ ;

    test_cie();   
    test_planck();   
    test_misc();   
    test_sliced();   
    return 0 ;
}
