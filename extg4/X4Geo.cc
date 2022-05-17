#include "X4Geo.hh"

/**
X4Geo::Translate
------------------

cf G4Opticks::tranlateGeometry 
trying to relocate here for reusablility 

**/

#include "X4PhysicalVolume.hh"
#include "GGeo.hh"
#include "PLOG.hh"

const plog::Severity X4Geo::LEVEL = PLOG::EnvLevel("X4Geo", "DEBUG"); 


GGeo* X4Geo::Translate(const G4VPhysicalVolume* top)  // static 
{
    //const char* keyspec = X4PhysicalVolume::Key(top) ;

    bool live = true ; 

    GGeo* gg = new GGeo( nullptr, live );   // picks up preexisting Opticks::Instance

    X4PhysicalVolume xtop(gg, top) ;

    gg->postDirectTranslation();    

    return gg ;  
} 

