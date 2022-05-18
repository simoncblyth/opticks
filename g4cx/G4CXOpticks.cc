
#include "U4GDML.h"
#include "X4Geo.hh"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"
#include "CSGOptiX.h"

#include "G4CXOpticks.hh"

G4CXOpticks::G4CXOpticks()
    :
    gg(nullptr),
    fd(nullptr), 
    cx(nullptr)
{
}

void G4CXOpticks::setGeometry(const char* gdmlpath)
{
    const G4VPhysicalVolume* world = U4GDML::Read(gdmlpath);
    setGeometry(world); 
}


/**
G4CXOpticks::setGeometry
--------------------------

HMM: instanciating CSGOptiX instanciates QSim for raygenmode other than zero 
and that needs the upload of QSim components first ?

**/


void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world)
{
    gg = X4Geo::Translate(world) ; 
    fd = CSG_GGeo_Convert::Translate(gg) ; 
    cx = CSGOptiX::Create(foundry); 
}

    
