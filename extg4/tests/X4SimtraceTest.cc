/**
X4SimtraceTest
======================

Used from script extg4/xxs0.sh 

**/
#include "OPTICKS_LOG.hh"
#include "X4Simtrace.hh"
#include "X4_GetSolid.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    X4Simtrace t ;

    // not encapsulated as do not want x4 to depend on PMTSim, only the tests 
    const G4VSolid* solid = X4_GetSolid(t.geom); 
    if( solid == nullptr ) LOG(fatal) << "failed to X4_GetSolid for geom " << t.geom  ; 
    assert( solid );   
    t.setSolid(solid); 


    t.simtrace(); 
    t.saveEvent() ; 

    return 0 ; 
}

