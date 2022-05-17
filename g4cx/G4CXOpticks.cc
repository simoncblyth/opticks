
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
    const G4VPhysicalVolume* world = U4GDML::Parse(gdmlpath);
    setGeometry(world); 
}

void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world)
{
    GGeo* ggeo = X4Geo::Translate(world) ; 
    foundry = CSG_GGeo_Convert::Translate(ggeo) ; 
    engine = new CSGOptiX(foundry); 
}

    
