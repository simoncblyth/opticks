/**
X4SimtraceTest
======================

Used from script extg4/x4t.sh 

Access to solids is not encapsulated as do not want x4 
to depend on PMTSim (which is JUNO specific). 
Only a few tests depend on PMTSim.

**/
#include "OPTICKS_LOG.hh"
#include "X4Simtrace.hh"
#include "X4_Get.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    X4Simtrace t ;

    const G4VSolid* solid = X4_Get::GetSolid(t.geom); 
    LOG_IF(fatal, solid == nullptr) << "failed to X4_GetSolid for geom " << t.geom  ; 
    assert( solid );   

    t.setSolid(solid); 
    t.simtrace(); 
    t.saveEvent() ; 

    return 0 ; 
}

