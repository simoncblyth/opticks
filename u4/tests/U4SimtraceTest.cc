/**
U4SimtraceTest.cc
===================

The below does simtrace scans of all unique solids in the geometry tree and
saves the placement transforms of all nodes into *base* with NP set_names
containing the solid names of all nodes.  This can be presented as 2D cross
sections of the geometry using U4PMTFastSimGeomTest.py 

Although in principal this should work for any geometry it is intended to
assist with debugging within small test geometries.  With large geometries it
will be very slow and write huge amounts of output. 


**/

#include "OPTICKS_LOG.hh"
#include "SPath.hh"
#include "U4VolumeMaker.hh"
#include "U4Simtrace.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const G4VPhysicalVolume* pv = U4VolumeMaker::PV();  // sensitive to GEOM envvar 
    const char* base = SPath::Resolve("$FOLD", DIRPATH ) ; 
    
    U4Simtrace t(pv); 
    t.scan(base);  

    return 0 ; 
}
