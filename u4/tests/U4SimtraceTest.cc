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

HMM: Not as general as this implies, because the gensteps pick a plane 
in which to collect intersects : so combining those only makes sense
when the transforms of the solids correspond. So it will work for the 
solids of a single PMT, but will not usually work across multiple PMTs. 

TODO: review the GPU simtrace approach and see if something similar 
could be done with G4Navigator  

**/

#include "G4VSolid.hh"

#include "OPTICKS_LOG.hh"
#include "stree.h"
#include "SEventConfig.hh"

#include "U4Tree.h"
#include "U4VolumeMaker.hh"

struct U4SimtraceTest
{
    stree st ; 
    U4Tree ut ; 

    U4SimtraceTest(const G4VPhysicalVolume* pv ); 
    void scan(const char* base ); 
}; 

inline U4SimtraceTest::U4SimtraceTest(const G4VPhysicalVolume* pv )
    :
    ut(&st, pv)   // instanciation of U4Tree populates the stree 
{
}

inline void U4SimtraceTest::scan(const char* base)
{
    ut.simtrace_scan(base) ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEventConfig::SetRGModeSimtrace(); 

    const G4VPhysicalVolume* pv = U4VolumeMaker::PV();  // sensitive to GEOM envvar 
    LOG(info) << " U4VolumeMaker::Desc() " << U4VolumeMaker::Desc() ; 
    
    U4SimtraceTest t(pv); 
    t.scan("$FOLD");  

    return 0 ; 
}


