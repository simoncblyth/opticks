
#include "PLOG.hh"

#include "scuda.h"
#include "sqat4.h"
#include "sframe.h"


#include "SSys.hh"
#include "SEvt.hh"
#include "SGeo.hh"
#include "SEventConfig.hh"
#include "SOpticksResource.hh"
#include "SFrameGenstep.hh"

#include "U4VolumeMaker.hh"

#include "SEventConfig.hh"
#include "U4GDML.h"
#include "U4Tree.h"

#include "CSGFoundry.h"
#include "CSG_GGeo_Convert.h"
#include "CSGOptiX.h"
#include "QSim.hh"

#include "G4CXOpticks.hh"



// OLD WORLD HEADERS STILL NEEDED UNTIL REJIG TRANSLATION 
#include "Opticks.hh"
#include "X4Geo.hh"
#include "GGeo.hh"



const plog::Severity G4CXOpticks::LEVEL = PLOG::EnvLevel("G4CXOpticks", "DEBUG"); 

G4CXOpticks* G4CXOpticks::INSTANCE = nullptr ; 
G4CXOpticks* G4CXOpticks::Get(){ return INSTANCE ; } 
// formerly instanciated when INSTANCE nullptr, but its better to require more care with when the instanciation is done

/**
G4CXOpticks::SetGeometry
--------------------------

Called for example from Detector framework LSExpDetectorConstruction_Opticks::Setup

**/

void G4CXOpticks::SetGeometry(const G4VPhysicalVolume* world)
{
    G4CXOpticks* g4cx = new G4CXOpticks ;
    g4cx->setGeometry(world); 
}

std::string G4CXOpticks::FormPath(const char* base, const char* rel )
{
    if( rel == nullptr ) rel = RELDIR ; 
    std::stringstream ss ;    
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   
    return dir ; 
}




void G4CXOpticks::Finalize() // static 
{
    LOG(LEVEL) << "placeholder mimic G4Opticks " ; 
}


G4CXOpticks::G4CXOpticks()
    :
    sd(nullptr),
    tr(nullptr),
    wd(nullptr),
    gg(nullptr),
    fd(nullptr), 
    cx(nullptr),
    qs(nullptr),
    se(nullptr)
{
    INSTANCE = this ; 
    LOG(LEVEL) << Desc() << std::endl << desc(); 
}

std::string G4CXOpticks::Desc() 
{
    return CSGOptiX::Desc() ; 
}

std::string G4CXOpticks::desc() const
{
    std::stringstream ss ; 
    ss << "G4CXOpticks::desc"
       << " sd " << sd 
       << " tr " << tr 
       << " wd " << wd 
       << " gg " << gg
       << " fd " << fd
       << " cx " << cx
       << " qs " << qs
       << " se " << se
       ;
    std::string s = ss.str(); 
    return s ; 
}

/**
G4CXOpticks::setSensor
---------------------------


**/

void G4CXOpticks::setSensor(const U4Sensor* sd_ )
{
    sd = sd_ ; 
}


/**
G4CXOpticks::setGeometry 
---------------------------

Without argument method geometry source depends on existance of envvars : SomeGDMLPath, CFBASE, GEOM

When not loading geometry from a CFBASE directory the CSGFoundry::save() method
is which saves the geometry to "$DefaultOutputDir/CSGFoundry" 
for use from python for example. 

**/

void G4CXOpticks::setGeometry()
{
    LOG(LEVEL) << " argumentless " ; 
    if(SSys::hasenvvar(SOpticksResource::SomeGDMLPath_))
    {
        LOG(LEVEL) << " SomeGDMLPath " ; 
        setGeometry(SOpticksResource::SomeGDMLPath()); 
        if(fd) fd->save(); 
    }
    else if(SSys::hasenvvar(SOpticksResource::CFBASE_))
    {
        LOG(LEVEL) << " CFBASE " ; 
        setGeometry(CSGFoundry::Load()); 
    }
    else if(SOpticksResource::CFBaseFromGEOM())
    {
        LOG(LEVEL) << " CFBASEFromGEOM " ; 
        setGeometry(CSGFoundry::Load()); 
    }
    else if(SSys::hasenvvar("GEOM"))
    {
        LOG(LEVEL) << " GEOM/U4VolumeMaker::PV " ; 
        setGeometry( U4VolumeMaker::PV() );  // this may load GDML using U4VolumeMaker::PVG if "GEOM"_GDMLPath is defined   
        if(fd) fd->save(); 
    }
    else
    {
        LOG(fatal) << " failed to setGeometry " ; 
        assert(0); 
    }
}

void G4CXOpticks::setGeometry(const char* gdmlpath)
{
    LOG(LEVEL) << " gdmlpath " << gdmlpath ; 
    const G4VPhysicalVolume* world = U4GDML::Read(gdmlpath);
    setGeometry(world); 
}

/**
G4CXOpticks::setGeometry
-------------------------

U4Tree/stree aspiring to become convertable to CSGFoundry and replace GGeo 

HMM: need a way to distingish between a de-natured world coming via GDML save/load  
and a live original world : as need to do things a bit differently in each case.

**/

void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
{
    LOG(LEVEL) << " G4VPhysicalVolume world " << world ; 
    assert(world); 
    wd = world ; 
    tr = U4Tree::Create(world) ;

#ifdef __APPLE__
    return ;  
#endif
   
    // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    Opticks::Configure("--gparts_transform_offset --allownokey" );  

    GGeo* gg_ = X4Geo::Translate(wd) ; 
    setGeometry(gg_); 
}
void G4CXOpticks::setGeometry(GGeo* gg_)
{
    LOG(LEVEL); 
    gg = gg_ ; 
    CSGFoundry* fd_ = CSG_GGeo_Convert::Translate(gg) ; 
    setGeometry(fd_); 

    // SAVING HERE(AFTER GGeo->CSGFoundry) TEMPORARILY : FOR INTEGRATION DEBUGGING  
    saveGeometry(); 
}
void G4CXOpticks::setGeometry(CSGFoundry* fd_)
{
    LOG(LEVEL) << " fd_ " << fd_ ; 
#ifdef __APPLE__
    return ; 
#endif
    fd = fd_ ; 

    se = new SEvt ; 
    se->setReldir("ALL"); 
    se->setGeo((SGeo*)fd);   // HMM: more general place for this hookup ?

    cx = CSGOptiX::Create(fd);   // uploads geometry to GPU 
    qs = cx->sim ; 
    LOG(LEVEL)  << " cx " << cx << " qs " << qs << " QSim::Get " << QSim::Get() ; 
}

void G4CXOpticks::render()
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif
    assert( cx ); 
    assert( SEventConfig::IsRGModeRender() ); 
    cx->render_snap() ; 
}

void G4CXOpticks::simulate()
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif

    LOG(LEVEL) << desc() ; 
    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimulate() ); 

    SEvt* sev = SEvt::Get();  assert(sev); 

    bool has_input_photon = sev->hasInputPhoton() ;
    if(has_input_photon)
    {
        const char* ipf = SEventConfig::InputPhotonFrame();
        sframe fr = fd->getFrame(ipf) ;  
        sev->setFrame(fr); 
    }

    unsigned num_genstep = sev->getNumGenstepFromGenstep(); 
    unsigned num_photon  = sev->getNumPhotonFromGenstep(); 

    LOG(LEVEL) 
        << "[ num_genstep " << num_genstep
        << " num_photon " << num_photon
        << " has_input_photon " << has_input_photon
        ;

    qs->simulate(); 

    sev->gather();   // downloads components configured by SEventConfig::CompMask 

    unsigned num_hit = sev->getNumHit() ; 
    bool is_undef = num_hit == SEvt::UNDEF ; 

    LOG(LEVEL) 
       << "] num_hit " << num_hit 
       << " is_undef " << is_undef 
       ;

    LOG(LEVEL) << " sev.descFull " << std::endl << sev->descFull() ; 

}

void G4CXOpticks::simtrace()
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif
    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimtrace() ); 
                           
    SEvt* sev = SEvt::Get();  assert(sev); 

    sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    sev->setFrame(fr);   // 

    cx->setFrame(fr);    
    // Q: why does cx need the frame ?

    qs->simtrace(); 
}
   


void G4CXOpticks::save() const 
{
    if(se == nullptr) return ; 
    se->save(); 

    LOG(LEVEL) << se->descPhoton() ; 
    LOG(LEVEL) << se->descLocalPhoton() ; 
    LOG(LEVEL) << se->descFramePhoton() ; 
}


void G4CXOpticks::saveGeometry() const
{
    //const char* def = SGeo::DefaultDir();   // huh giving null 
    const char* def = SEventConfig::OutFold() ; 
    std::cout << "G4CXOpticks::saveGeometry def [" << ( def ? def : "-" ) << std::endl ; 
    saveGeometry(def) ; 
}
void G4CXOpticks::saveGeometry(const char* base, const char* rel) const
{
    std::string dir = FormPath(base, rel); 
    saveGeometry_(dir.c_str()); 
}
void G4CXOpticks::saveGeometry_(const char* dir_) const
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    std::cout << "[ G4CXOpticks::saveGeometry_ " << ( dir ? dir : "-" ) << std::endl ; 
    const stree* st = tr ? tr->st : nullptr ; 
    if(st) st->save(dir) ;   
    if(fd) fd->save(dir) ; 
    if(gg) gg->save_to_dir(dir) ; 
    if(wd) U4GDML::Write(wd, dir, "origin.gdml" );
    std::cout << "] G4CXOpticks::saveGeometry_ " << ( dir ? dir : "-" ) << std::endl ; 
}


