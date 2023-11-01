/**
U4Custom4Test.cc
=================

WITH_SLOG is defined already within Opticks,
but its not defined for the C4 build so CUSTOM4_LOG_ gives missing symbols. 
Dont want to complexify C4. Solution is to get headeronly SLOG to work. 

**/
#include <iostream>
#include "G4Track.hh"
#include "G4OpBoundaryProcess.hh"
#include "OPTICKS_LOG.hh"

#ifdef WITH_CUSTOM4


#include "C4OpBoundaryProcess.hh"
#include "C4CustomART.h"

// mock Accessor standin for junosw PMTAccessor
#include "U4PMTAccessor.h"

#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_CUSTOM4
    U4PMTAccessor* pmt = new U4PMTAccessor ; 
    C4IPMTAccessor* ipmt = pmt ; 
    C4OpBoundaryProcess* proc = new C4OpBoundaryProcess(pmt) ;     
    std::cout << " proc " << std::hex << proc << std::dec << std::endl ;
#endif


    std::cout 
#ifdef WITH_CUSTOM4
        << "WITH_CUSTOM4" 
#else
        << "not-WITH_CUSTOM4" 
#endif
        << std::endl 
        ; 


    return 0 ; 
}
