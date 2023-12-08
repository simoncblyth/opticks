
#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "scerenkov.h"

struct scerenkov__test
{
    static void t0();  
    static void t1();  
    static int main();  
};

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
    scerenkov& gs = (scerenkov&)_gs ;  // warning: dereferencing type-punned pointer will break strict-aliasing rules
    gs.numphoton = 101 ; 
    std::cout << _gs.desc() << std::endl  ; 
}


int scerenkov__test::main()
{
    t1(); 
    return 0 ; 
}

int main(){ return scerenkov__test::main(); }


