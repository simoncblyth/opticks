/**
U4SimulateTest.cc  (HMM: maybe rename U4AppTest.cc)
====================================================

All the Geant4 setup happens in U4App::Create from U4App.h

**/

#include "U4App.h"    
#include "OPTICKS_LOG.hh"

#if defined(WITH_PMTSIM) && defined(PMTSIM_STANDALONE)
#include "J_PMTSIM_LOG.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#if defined(WITH_PMTSIM) && defined(PMTSIM_STANDALONE)
    J_PMTSIM_LOG_(0); 
    std::cout << "main: WITH_PMTSIM and PMTSIM_STANDALONE  " << std::endl ; 
#else
    std::cout << "main: not-( WITH_PMTSIM and PMTSIM_STANDALONE )  " << std::endl ; 
#endif

    U4App::Main(); 

    return 0 ; 
}

