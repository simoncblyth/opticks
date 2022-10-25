#include "X4Geo.hh"

/**
X4Geo::Translate
------------------

cf G4Opticks::tranlateGeometry 
trying to relocate here for reusablility 

**/

#include "X4PhysicalVolume.hh"
#include "GGeo.hh"
#include "SLOG.hh"

const plog::Severity X4Geo::LEVEL = SLOG::EnvLevel("X4Geo", "DEBUG"); 


GGeo* X4Geo::Translate(const G4VPhysicalVolume* top)  // static 
{
    bool live = true ; 

    GGeo* gg = new GGeo( nullptr, live );   // picks up preexisting Opticks::Instance

    X4PhysicalVolume xtop(gg, top) ;  // lots of heavy lifting translation in here 

    gg->postDirectTranslation();    

    return gg ;  
} 

