
#include "SEventConfig.hh"
#include "U4GDML.h"
#include "X4Geo.hh"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"
#include "CSGOptiX.h"

#include "G4CXOpticks.hh"

G4CXOpticks::G4CXOpticks()
    :
    wd(nullptr),
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
void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world)
{
    wd = world ; 
    GGeo* gg_ = X4Geo::Translate(wd) ; 
    setGeometry(gg_); 
}
void G4CXOpticks::setGeometry(const GGeo* gg_)
{
    gg = gg_ ; 
    CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ; 
    setGeometry(fd_); 
}
void G4CXOpticks::setGeometry(CSGFoundry* fd_)
{
    fd = fd_ ; 
    cx = CSGOptiX::Create(fd); 
}

void G4CXOpticks::snap()
{
    assert( cx ); 
    assert( SEventConfig::IsRGModeRender() ); 
    cx->render() ; 
    cx->snap("/tmp/out.jpg"); 
}

    
