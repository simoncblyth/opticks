/**
QOpticalTest.cc
===================

TODO: combine optical with bnd as they are so closely related it 
makes no sense to treat them separately 

**/
#include <cuda_runtime.h>
#include "scuda.h"
#include "NP.hh"

#include "QOptical.hh"

int main(int argc, char** argv)
{
    const char* BASE = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard" ; 
    NP* optical = NP::Load(BASE, "optical.npy") ; 
    if( optical == nullptr ) return 1 ; 
   
    std::cout 
        << " optical " << ( optical ? optical->sstr() : "-" )
        << std::endl 
        ;

    QOptical q_optical(optical) ; 
    std::cout << q_optical.desc() << std::endl ; 

    q_optical.check(); 

    cudaDeviceSynchronize(); 

    return 0 ; 
}
