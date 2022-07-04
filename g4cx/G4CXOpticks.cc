
#include "PLOG.hh"
#include "SEventConfig.hh"
#include "U4GDML.h"
#include "X4Geo.hh"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"
#include "CSGOptiX.h"

#include "QSim.hh"

#include "G4CXOpticks.hh"

const plog::Severity G4CXOpticks::LEVEL = PLOG::EnvLevel("G4CXOpticks", "DEBUG"); 

std::string G4CXOpticks::Desc() 
{
    return CSGOptiX::Desc() ; 
}


G4CXOpticks::G4CXOpticks()
    :
    wd(nullptr),
    gg(nullptr),
    fd(nullptr), 
    cx(nullptr),
    qs(nullptr)
{
}


std::string G4CXOpticks::desc() const
{
    std::stringstream ss ; 
    ss << "G4CXOpticks::desc"
       << " wd " << wd
       << " gg " << gg
       << " fd " << fd
       << " cx " << cx
       << " qs " << qs
       ;
    std::string s = ss.str(); 
    return s ; 
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
    qs = cx->sim ; 
    LOG(LEVEL)  << " cx " << cx << " qs " << qs << " QSim::Get " << QSim::Get() ; 
}

void G4CXOpticks::render()
{
    assert( cx ); 
    assert( SEventConfig::IsRGModeRender() ); 
    cx->render_snap() ; 
}

void G4CXOpticks::simulate()
{
    LOG(LEVEL) << desc() ; 
    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimulate() ); 
    qs->simulate(); 
}

void G4CXOpticks::simtrace()
{
    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimtrace() ); 
    qs->simtrace(); 
}


    
