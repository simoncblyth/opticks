#include <cstdlib>
#include <cassert>
#include <iostream>

#include "U4Random.hh"

void test_basics(U4Random* rnd)
{
    std::cout << "rnd.m_seqpath " << rnd->m_seqpath << std::endl ; 

    std::cout << " rand.dump asis : STANDARD G4UniformRand " << std::endl ; 
    rnd->dump(); 
    rnd->setSequenceIndex(0); 
    std::cout << " rand.dump after U4Random::setSequenceIndex(0) : USING PRECOOKED RANDOMS " << std::endl ; 
    rnd->dump(); 
    rnd->setSequenceIndex(-1); 
    std::cout << " rand.dump after U4Random::setSequenceIndex(-1) : BACK TO STANDARD RANDOMS " << std::endl ; 
    rnd->dump(); 
}

int main()
{
    U4Random* rnd = new U4Random ; 

    test_basics(rnd); 

    return 0 ; 
}

