/**
U4NavigatorTest.cc
====================

This fails with fHistory.GetTopVolume() giving nullptr
however U4Navigator::Check() works from U4App.h
following Geant4 run manager setup.

So are missing some geometry and/or tracking setup here,
probably the voxelization happens as part of the full
setup or some such.

This means that G4Navigator based simtracing needs to be added
as a sibling alternative method to U4App::BeamOn rather
than implementing in a separate binary as the limited G4VSolid based
simtracing is currently done.

**/

#include "OPTICKS_LOG.hh"
#include "U4VolumeMaker.hh"
#include "U4Navigator.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const G4VPhysicalVolume* pv = U4VolumeMaker::PV();  // sensitive to GEOM envvar
    if(pv == nullptr) return 0 ;

    G4ThreeVector ori(0.,0.,0.);
    G4ThreeVector dir(0.,0.,1.);

    G4double dist = U4Navigator::Distance( ori, dir );
    LOG(info) << " dist " << dist  ;

    return 0 ;
}

