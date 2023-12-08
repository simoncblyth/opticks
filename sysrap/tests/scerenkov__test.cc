
#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "scerenkov.h"

struct scerenkov__test
{
#ifdef WITH_DEREF_WARNNG
    static void t0();  
    static void t1();  
#endif
    static void t2();  
    static void t3();  
    static int main();  
};

#ifdef WITH_DEREF_WARNNG
void scerenkov__test::t0()
{
    quad6 _gs ;
    _gs.zero() ; 
    scerenkov& gs = (scerenkov&)_gs ;  // warning: dereferencing type-punned pointer will break strict-aliasing rules
    gs.numphoton = 101 ; 
    std::cout << _gs.desc() << std::endl  ; 
}
void scerenkov__test::t1()
{
    quad6 _gs ;
    _gs.zero() ; 
    scerenkov& gs = reinterpret_cast<scerenkov&>(_gs) ; 
    gs.numphoton = 101 ; 
    std::cout << _gs.desc() << std::endl  ; 
}
#endif

void scerenkov__test::t2()
{
    quad6 _gs ;
    _gs.zero() ; 
    scerenkov* gs = reinterpret_cast<scerenkov*>(&_gs) ; 
    gs->numphoton = 101 ; 
    std::cout << _gs.desc() << std::endl  ; 
}
void scerenkov__test::t3()
{
    quad6 _gs ;
    _gs.zero() ; 
    scerenkov* gs = (scerenkov*)(&_gs) ; 
    gs->numphoton = 101 ; 
    std::cout << _gs.desc() << std::endl  ; 
}


int scerenkov__test::main()
{
    t2(); 
    t3(); 
    return 0 ; 
}

int main(){ return scerenkov__test::main(); }


