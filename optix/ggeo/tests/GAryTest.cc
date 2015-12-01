#include "GAry.hh"
#include "assert.h"

typedef GAry<float> A ;

void test_planck()
{

   A* a = A::linspace(500, 300., 800.);

   A* b = A::planck_spectral_radiance( a, 6500.f ); 

   b->Summary();

   b->save("/tmp/ary_planck.npy");
}


void test_misc()
{
    A* a = A::from_constant(10, 1.f) ;
    a->Summary("a");

    A* c = a->cumsum();
    c->Summary("a->cumsum()");

    A* u = A::urandom();
    u->Summary("u");
    u->save("/tmp/urandom.npy");
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
    //test_sliced();   
    test_planck();   
    return 0 ;
}
