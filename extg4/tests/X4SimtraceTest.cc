/**
X4SimtraceTest
======================

Used from script extg4/x4t.sh 

Access to solids is not encapsulated as do not want x4 
to depend on PMTSim (which is JUNO specific). 
Only a few tests depend on PMTSim.

**/
#include "OPTICKS_LOG.hh"

#include "ssys.h"
#include "X4Simtrace.hh"
#include "X4_Get.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* geom = ssys::getenvvar("X4SimtraceTest_GEOM", "nmskSolidMaskTail") ; 
    const G4VSolid* solid = X4_Get::GetSolid(geom); 
    LOG_IF(fatal, solid == nullptr) << "failed to X4_GetSolid for geom " << geom  ; 
    assert( solid );   

    X4Simtrace::Scan(solid); 


    return 0 ; 
}

