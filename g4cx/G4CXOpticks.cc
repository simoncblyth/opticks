
#include "U4GDML.h"
#include "X4Geo.hh"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"
#include "CSGOptiX.h"

#include "G4CXOpticks.hh"

G4CXOpticks::G4CXOpticks()
    :
    foundry(nullptr), 
    engine(nullptr)
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
    GGeo* ggeo = X4Geo::Translate(world) ; 
    foundry = CSG_GGeo_Convert::Translate(ggeo) ; 
    engine = new CSGOptiX(foundry); 
}

    
