#include <cstdlib>
#include <cassert>
#include <iostream>

#include "OPTICKS_LOG.hh"
#include "U4Random.hh"

void test_basics(U4Random* rnd)
{
    LOG(info) << "rnd.m_seqpath " << rnd->m_seqpath ; 

    LOG(info) << " rand.dump asis : STANDARD G4UniformRand " ; 
    rnd->dump(); 

    LOG(info) << " rand.dump after U4Random::setSequenceIndex(0) : USING PRECOOKED RANDOMS " ; 
    rnd->setSequenceIndex(0); 
    rnd->dump(); 


    LOG(info) << " rand.dump after U4Random::setSequenceIndex(-1) : BACK TO STANDARD RANDOMS " ; 
    rnd->setSequenceIndex(-1); 
    rnd->dump(); 

    LOG(info) << " rand.dump after U4Random::setSequenceIndex(0) : USING PRECOOKED RANDOMS AGAIN " ; 
    rnd->setSequenceIndex(0); 
    rnd->dump(); 



}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    U4Random* rnd = new U4Random ; 

    test_basics(rnd); 

    return 0 ; 
}

