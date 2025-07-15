/**
G4CX_U4TreeCreateCSGFoundryTest.cc
====================================

1. creates Geant4 pv with U4VolumeMaker::PV which is configured via GEOM envvar,
   see also U4SolidMaker::Make for implemented names and to add more
2. applies U4Tree::Create populating an stree
3. invokes CSGFoundry::CreateFromSim which imports the stree
4. saves the CSGFoundry to $FOLD

**/

#include "SLOG.hh"
#include "SSim.hh"
#include "U4Tree.h"
#include "CSGFoundry.h"
#include "U4VolumeMaker.hh"

struct G4CX_U4TreeCreateCSGFoundryTest
{
    const G4VPhysicalVolume* world ;
    SSim* sim ;
    stree*  st ;
    U4Tree* tr ;
    CSGFoundry* fd ;

    G4CX_U4TreeCreateCSGFoundryTest();
};

inline G4CX_U4TreeCreateCSGFoundryTest::G4CX_U4TreeCreateCSGFoundryTest()
    :
    world(U4VolumeMaker::PV()),
    sim(SSim::CreateOrReuse()),
    st(sim->get_tree()),
    tr(U4Tree::Create(st,world,nullptr)),
    fd(nullptr)
{
    sim->initSceneFromTree();
    fd = CSGFoundry::CreateFromSim();
    fd->save("$FOLD");
}

int main()
{
    G4CX_U4TreeCreateCSGFoundryTest t ;
    return 0 ;
}
