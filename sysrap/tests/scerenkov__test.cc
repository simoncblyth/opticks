
#include <iostream>

#include "scuda.h"
#include "squad.h"
#include "scerenkov.h"

int main()
{
    quad6 _gs ;
    _gs.zero() ; 

    scerenkov& gs = (scerenkov&)_gs ;  // warning: dereferencing type-punned pointer will break strict-aliasing rules

    gs.numphoton = 101 ; 


    std::cout << _gs.desc() << std::endl  ; 

    return 0 ; 
}
