#include <iostream>
#include <cstdlib>

#include "Randomize.hh"
#include "SUniformRand.h"
typedef SUniformRand<CLHEP::HepRandom> U4UniformRand ; 

const char* FOLD = getenv("FOLD"); 

int main(int argc, char** argv)
{
    NP* uu = U4UniformRand::Get(1000) ;  
    std::cout << " uu " << uu->repr<double>() << std::endl ; 
    uu->save(FOLD, "uu.npy"); 

    return 0 ; 
}
