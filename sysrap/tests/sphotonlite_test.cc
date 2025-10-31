/**
sphotonlite_test.cc
====================

::

     ~/o/sysrap/tests/sphotonlite_test.sh

**/

#include <iostream>
#include <array>
#include <bitset>

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sphotonlite.h"
#include "NPFold.h"
#include "OpticksPhoton.h"
#include "ssys.h"

struct sphotonlite_test
{
    static int ctor();
    static int demoarray();
    static int main();
};


int sphotonlite_test::ctor()
{
    unsigned fm = EFFICIENCY_COLLECT ;
    unsigned id = 12345 ;

    float lposcost_0 = 0.5f ; 
    float lposfphi_0 = 0.6f ; 

    sphoton p = {} ;
    p.flagmask = fm ;
    p.identity = id ;
    p.time = 101.101f ; 

    sphotonlite l = {} ;
    l.init(p.identity, p.time, p.flagmask );
    l.set_lpos( lposcost_0, lposfphi_0 );

    float lposcost_1(0.f) ; 
    float lposfphi_1(0.f) ; 
    l.get_lpos( lposcost_1, lposfphi_1 ); 

    assert( l.flagmask == fm );

    std::cout << "p\n" << p.desc() << "\n" ; 
    std::cout << "l\n" << l.desc() << "\n" ; 

    return 0 ;
}

int sphotonlite_test::demoarray()
{
    NP* l = sphotonlite::make_demoarray(10);
    l->save("$FOLD/demoarray.npy"); 
    return 0 ;
}


int sphotonlite_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","make_demoarray") ;
    bool ALL = 0 == strcmp(TEST, "ALL");

    int rc = 0 ;
    if(ALL||0==strcmp(TEST, "ctor"))       rc += ctor();
    if(ALL||0==strcmp(TEST, "demoarray"))  rc += demoarray();

    return rc ;
}

int main()
{
    return sphotonlite_test::main();
}

