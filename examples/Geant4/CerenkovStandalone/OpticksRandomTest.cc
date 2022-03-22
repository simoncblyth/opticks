#include <cstdlib>
#include <cassert>
#include <iostream>

#include "OpticksRandom.hh"

void test_basics(OpticksRandom* rnd)
{
    std::cout << "rnd.m_seqpath " << rnd->m_seqpath << std::endl ; 

    std::cout << " rand.dump asis : STANDARD G4UniformRand " << std::endl ; 
    rnd->dump(); 
    rnd->setSequenceIndex(0); 
    std::cout << " rand.dump after OpticksRandom::setSequenceIndex(0) : USING PRECOOKED RANDOMS " << std::endl ; 
    rnd->dump(); 
    rnd->setSequenceIndex(-1); 
    std::cout << " rand.dump after OpticksRandom::setSequenceIndex(-1) : BACK TO STANDARD RANDOMS " << std::endl ; 
    rnd->dump(); 
}

int main()
{
    OpticksRandom* rnd = new OpticksRandom ; 
    rnd->m_flat_debug = true ; 

    test_basics(rnd); 

    return 0 ; 
}

