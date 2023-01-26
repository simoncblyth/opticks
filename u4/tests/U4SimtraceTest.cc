/**
U4SimtraceTest.cc
===================

Usually invoked from U4SimtraceTest.sh

This does simtrace scans of all unique solids in a Geant4 geometry tree and
saves the placement transforms of all nodes into *base* directory with NP set_names
containing the solid names of all nodes.  These folders of .npy files 
can be presented as 2D cross sections of the geometry using U4SimtraceTest.py 
as orchestrated by U4SimtraceTest.sh

Although in principal this should work for any geometry it is intended to
assist with debugging within small test geometries.  With large geometries it
will be very slow and write huge amounts of output. 

**/

#include "OPTICKS_LOG.hh"
#include "SPath.hh"

#include "snd.h"
std::vector<snd> snd::node  = {} ; 
std::vector<spa> snd::param = {} ; 
std::vector<sxf> snd::xform = {} ; 
std::vector<sbb> snd::aabb  = {} ; 
// HMM: how to avoid ? 



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
