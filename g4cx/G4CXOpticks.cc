
#include <csignal>
#include "SLOG.hh"


#include "scuda.h"
#include "sqat4.h"
#include "sframe.h"

#include "ssys.h"
#include "smeta.h"

#include "SEvt.hh"
#include "SSim.hh"
#include "SGeo.hh"
#include "SOpticksResource.hh"
#include "SFrameGenstep.hh"

#include "U4VolumeMaker.hh"

#include "SEventConfig.hh"
#include "U4GDML.h"
#include "U4Tree.h"

#include "CSGFoundry.h"

#include "CSGOptiX.h"
#include "QSim.hh"

#include "G4CXOpticks.hh"

#ifdef WITH_CUSTOM4
#include "C4Version.h"
#endif


const plog::Severity G4CXOpticks::LEVEL = SLOG::EnvLevel("G4CXOpticks", "DEBUG"); 

U4SensorIdentifier* G4CXOpticks::SensorIdentifier = nullptr ; 
void G4CXOpticks::SetSensorIdentifier( U4SensorIdentifier* sid ){ SensorIdentifier = sid ; }  // static 


G4CXOpticks* G4CXOpticks::INSTANCE = nullptr ; 
G4CXOpticks* G4CXOpticks::Get(){ return INSTANCE ; } 

/**
G4CXOpticks::SetGeometry
--------------------------

Called for example from Detector framework LSExpDetectorConstruction_Opticks::Setup

**/

G4CXOpticks* G4CXOpticks::SetGeometry()
{
    G4CXOpticks* g4cx = new G4CXOpticks ;
    g4cx->setGeometry(); 
    return g4cx ; 
}
G4CXOpticks* G4CXOpticks::SetGeometry(const G4VPhysicalVolume* world)
{
    G4CXOpticks* g4cx = new G4CXOpticks ;
    g4cx->setGeometry(world); 
    return g4cx ; 
}

void G4CXOpticks::Finalize() // static 
{
    LOG(LEVEL); 
}

#ifdef __APPLE__
bool G4CXOpticks::NoGPU = true ; 
#else
bool G4CXOpticks::NoGPU = false ; 
#endif
void G4CXOpticks::SetNoGPU(bool no_gpu){ NoGPU = no_gpu ; }  
bool G4CXOpticks::IsNoGPU(){  return NoGPU ; }  


/**
G4CXOpticks::G4CXOpticks
--------------------------

HMM: sim in GX ctor seems out of place, shouldnt that be coming from CSGFoundry ?

* part of reason is need to get stree from SSim during conversion before CSGFoundry ctor 
* maybe CSGFoundry should be adopting any preexisting SSim 

* HMM: is this is an argument for SSim to live with SEvt rather than with the geometry ?
* NO : constant nature of SSim makes saving it with the geometry make a lot of sense, 
  so best to stay with the geometry

**/

G4CXOpticks::G4CXOpticks()
    :
    sim(SSim::CreateOrReuse()),   
    tr(nullptr),
    wd(nullptr),
    fd(nullptr), 
    cx(nullptr),
    qs(nullptr),
    t0(schrono::stamp())
{
    init();
}

void G4CXOpticks::init()
{
    INSTANCE = this ; 
    LOG(LEVEL) << Desc() << std::endl << desc(); 
}

G4CXOpticks::~G4CXOpticks()
{
    schrono::TP t1 = schrono::stamp(); 
    double dt = schrono::duration(t0, t1 );
    LOG(LEVEL) << "lifetime " << std::setw(10) << std::fixed << std::setprecision(3) << dt << " s " ; 
}


std::string G4CXOpticks::Desc() 
{
    return CSGOptiX::Desc() ; 
}

std::string G4CXOpticks::desc() const
{
    std::stringstream ss ; 
    ss << "G4CXOpticks::desc"
       << " sim " << ( sim ? "Y" : "N" )
       << " tr " << ( tr ? "Y" : "N" ) 
       << " wd " << ( wd ? "Y" : "N" )
       << " fd " << ( fd ? "Y" : "N" )
       << " cx " << ( cx ? "Y" : "N" )
       << " qs " << ( qs ? "Y" : "N" )
       ;
    std::string s = ss.str(); 
    return s ; 
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

    if(ssys::hasenv_(SOpticksResource::OpticksGDMLPath_))
    {
        LOG(LEVEL) << " OpticksGDMLPath " ; 
        setGeometry(SOpticksResource::OpticksGDMLPath()); 
    }
    else if(ssys::hasenv_(SOpticksResource::SomeGDMLPath_))
    {
        LOG(LEVEL) << " SomeGDMLPath " ; 
        setGeometry(SOpticksResource::SomeGDMLPath()); 
    }
    else if(ssys::hasenv_(SOpticksResource::CFBASE_))
    {
        LOG(LEVEL) << " CFBASE " ; 
        setGeometry(CSGFoundry::Load()); 
    }
    else if(SOpticksResource::CFBaseFromGEOM())
    {
        LOG(LEVEL) << "[ CFBASEFromGEOM " ; 

        LOG(LEVEL) << "[ CSGFoundry::Load " ; 
        CSGFoundry* cf = CSGFoundry::Load() ;
        LOG(LEVEL) << "] CSGFoundry::Load " ; 

        LOG(LEVEL) << "[ setGeometry(cf)  " ; 
        setGeometry(cf); 
        LOG(LEVEL) << "] setGeometry(cf)  " ; 

        LOG(LEVEL) << "] CFBASEFromGEOM " ; 
    }
    else if(SOpticksResource::GDMLPathFromGEOM())
    {
        // may load GDML directly if "${GEOM}_GDMLPathFromGEOM" is defined
        LOG(LEVEL) << "[ GDMLPathFromGEOM " ; 
        setGeometry(SOpticksResource::GDMLPathFromGEOM()) ; 
        LOG(LEVEL) << "] GDMLPathFromGEOM " ; 
    }
    else if(ssys::hasenv_("GEOM"))
    {
       // this may load GDML using U4VolumeMaker::PVG if "${GEOM}_GDMLPath" is defined   
        LOG(LEVEL) << " GEOM/U4VolumeMaker::PV " ; 
        setGeometry( U4VolumeMaker::PV() );  
    }
    else
    {
        LOG(fatal) << " failed to setGeometry " ; 
        assert(0); 
    }
}

void G4CXOpticks::setGeometry(const char* gdmlpath)
{
    LOG(LEVEL) << " gdmlpath [" << gdmlpath << "]" ;
    const G4VPhysicalVolume* world = U4GDML::Read(gdmlpath);

    setGeometry(world); 
}

/**
G4CXOpticks::setGeometry
-------------------------

U4Tree/stree+SSim replacing the former GGeo+X4+.. packages 

HMM: need a way to distingish between a re-animated world coming via GDML save/load  
and a live original world : as need to do things a bit differently in each case.

* could determine this by noticing the lack of SensitiveDetectors in GDML re-animated world,
  (but thats kinda heavy way to determine one bit) OR by passing a signal along with the 
  world to show that it has been re-animated  


Q: Is stree.h/st in actual use yet ? Where ? What parts of GGeo does that replace ?
A: YES, stree.h is already playing a vital role as *tree* member of CSG_GGeo/CSG_GGeo_Convert.cc 
   see CSG_GGeo_Convert::addInstances where the sensor identifier gets incorporated 
   into the instances


**/


void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
{
    LOG(LEVEL) << "[ G4VPhysicalVolume world " << world ; 
    assert(world); 
    wd = world ;

    assert(sim && "sim instance should have been grabbed/created in ctor" ); 
    stree* st = sim->get_tree(); 

    tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    LOG(LEVEL) << "Completed U4Tree::Create " ; 

    CSGFoundry* fd_ = CSGFoundry::CreateFromSim() ; // adopts SSim::INSTANCE  
    setGeometry(fd_); 

    LOG(LEVEL) << "] G4VPhysicalVolume world " << world ; 
}



/**
G4CXOpticks::setGeometry
---------------------------

Prior to CSGOptiX::Create the SEvt instance is created. 

Q: is there a more general place for SEvt hookup ?
A: SSim could hold the SEvt together with stree ?

But SEvt feels like it should be separate, 
as the SSim focus is initialization and SEvt focus is post-init.  

**/

const char* G4CXOpticks::setGeometry_saveGeometry = ssys::getenvvar("G4CXOpticks__setGeometry_saveGeometry") ;
void G4CXOpticks::setGeometry(CSGFoundry* fd_)
{
    setGeometry_(fd_); 
    setupFrame();    // EXPT: MOVED HERE TO INITIALIZATION
}


/**
G4CXOpticks::setGeometry_
---------------------------

Has side-effect of calling SSim::serialize which 
adds the stree and extra subfolds to the SSim subfold. 

Unclear where exactly SSim::serialize needs to happen 
for sure it must be after U4Tree::Create and before QSim 
instanciation within CSGOptiX::Create 

Note would be better to encapsulate this SSim handling 
within U4Tree.h but reluctant currently as U4Tree.h is  
header only and SSim.hh is not. 

Q: Is the difficulty only due to the need to defer for the extras ? 
 
**/

void G4CXOpticks::setGeometry_(CSGFoundry* fd_)
{
    fd = fd_ ; 
    LOG(LEVEL) << "[ fd " << fd ; 

    init_SEvt(); 

    if(NoGPU == false)
    {
        LOG(LEVEL) << "[ CSGOptiX::Create " ;  
        cx = CSGOptiX::Create(fd);   // uploads geometry to GPU 
        LOG(LEVEL) << "] CSGOptiX::Create " ;  
    }
    else
    {
        LOG(LEVEL) << " skip CSGOptiX::Create as NoGPU has been set " ;  
    }

    qs = cx ? cx->sim : nullptr ; 
   
    QSim* qsg = QSim::Get()  ;  

    LOG(LEVEL)  
        << " cx " << ( cx ? "Y" : "N" ) 
        << " qs " << ( qs ? "Y" : "N" )
        << " QSim::Get " << ( qsg ? "Y" : "N" ) 
        ; 

    LOG(LEVEL) << "] fd " << fd ; 


    // try moving this later, so can save things from cx and qs for debug purposes
    if( setGeometry_saveGeometry != nullptr )
    {
        LOG(info) << "[ G4CXOpticks__setGeometry_saveGeometry " ;  
        saveGeometry(setGeometry_saveGeometry); 
        LOG(info) << "] G4CXOpticks__setGeometry_saveGeometry " ;  
    }
}

/**
G4CXOpticks::init_SEvt
------------------------

Typically this instanciates the SEvt::EGPU instance once only 
at initialization, invoked from `G4CXOpticks::setGeometry_`

**/

void G4CXOpticks::init_SEvt()
{
    sim->serialize() ;  
    SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ; 
    sev->setGeo((SGeo*)fd);   
    smeta::Collect(sev->meta, "G4CXOpticks::init_SEvt"); 

    std::string gm = sim->getGPUMeta() ; 
    sev->setMeta("GPUMeta", gm.c_str() );  // set CUDA_VISIBLE_DEVICES to control 

#ifdef WITH_CUSTOM4
    std::string c4 = C4Version::Version();
    sev->setMeta("C4Version", c4.c_str()); 
#else
    sev->setMeta("C4Version", "NOT-WITH_CUSTOM4" ); 
#endif

}




/**
G4CXOpticks::setupFrame
-------------------------

The frame used depends on envvars INST, MOI, OPTICKS_INPUT_PHOTON_FRAME 
it comprises : fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 

This setupFrame was formerly called from G4CXOpticks::simulate and G4CXOpticks::simtrace
it is now moved to G4CXOpticks::setGeometry to facilitate transformation 
of input photons. 

Q: why is the frame needed ?
A: cx rendering viewpoint, input photon frame and the simtrace genstep grid 
   are all based on the frame center, extent and transforms 

Q: Given the sframe and SEvt are from sysrap it feels too high level to do this here, 
   should be at CSG or sysrap level perhaps ? 
   And then CSGOptix could grab the SEvt frame at its initialization. 

**/

void G4CXOpticks::setupFrame()
{
    // TODO: see CSGFoundry::AfterLoadOrCreate for auto frame hookup

    sframe fr = fd->getFrameE() ; 
    LOG(LEVEL) << fr ; 

    SEvt::SetFrame(fr) ; 

    if(cx) cx->setFrame(fr);  
}



std::string G4CXOpticks::descSimulate() const
{
    SEvt* sev = SEvt::Get_EGPU() ;  
    assert(sev); 

    int sev_index = sev->getIndex() ;
    unsigned num_genstep = sev->getNumGenstepFromGenstep(); 
    unsigned num_photon  = sev->getNumPhotonCollected(); 
    unsigned num_hit = sev->getNumHit() ; 
    bool is_undef = num_hit == SEvt::UNDEF ; 

    std::stringstream ss ;  
    ss << "G4CXOpticks::descSimulate"
       << " sev_instance " << sev->instance 
       << " sev_index " << sev_index 
       << " num_genstep " << num_genstep 
       << " num_photon " << num_photon 
       << " num_hit " << num_hit 
       << " num_hit.is_undef " << ( is_undef ? "YES" : "NO " )
       << " sev.brief " << sev->brief()  
       ;

    std::string str = ss.str(); 
    return str ; 
}

/**
G4CXOpticks::simulate
-----------------------

GPU launch doing generation and simulation done here 

**/


void G4CXOpticks::simulate(int eventID)
{
    LOG_IF(fatal, NoGPU) << "NoGPU SKIP" ; 
    if(NoGPU) return ; 

    LOG(LEVEL) << "[" ; 
    LOG(LEVEL) << desc() ; 

    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimulate() ); 

    qs->simulate(eventID);   
    LOG(LEVEL) << "]" ; 
}

void G4CXOpticks::simtrace(int eventID)
{
    LOG_IF(fatal, NoGPU) << "NoGPU SKIP" ; 
    if(NoGPU) return ; 

    LOG(LEVEL) << "[" ; 
    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimtrace() ); 

    qs->simtrace(eventID); 
    LOG(LEVEL) << "]" ; 
}

void G4CXOpticks::render()
{
    LOG_IF(fatal, NoGPU) << "NoGPU SKIP" ; 
    if(NoGPU) return ; 

    LOG(LEVEL) << "[" ; 
    assert( cx ); 
    assert( SEventConfig::IsRGModeRender() ); 
    cx->render() ; 
    LOG(LEVEL) << "]" ; 
}






/**
G4CXOpticks::saveGeometry
---------------------------

This is called from G4CXOpticks::setGeometry after geometry translation when::

    export G4CXOpticks__setGeometry_saveGeometry=1
    export G4CXOpticks=INFO       # to see the directory path 

What is saved::

    N[blyth@localhost ~]$ l /tmp/blyth/opticks/GEOM/ntds3/G4CXOpticks/
    total 41012
        0 drwxr-xr-x.  4 blyth blyth      109 Oct  5 18:21 .
        4 -rw-rw-r--.  1 blyth blyth      201 Oct  5 18:21 origin_gdxml_report.txt
    20504 -rw-rw-r--.  1 blyth blyth 20992917 Oct  5 18:21 origin.gdml
    20504 -rw-rw-r--.  1 blyth blyth 20994470 Oct  5 18:21 origin_raw.gdml
        0 drwxrwxr-x. 15 blyth blyth      273 Oct  5 18:21 GGeo
        0 drwxr-xr-x.  3 blyth blyth      190 Oct  5 18:21 CSGFoundry
        0 drwxr-xr-x.  3 blyth blyth       25 Oct  5 18:21 ..
    N[blyth@localhost ~]$ 

Grab that locally::

    GEOM      ## set the envvar 
    GEOM get  ## grab the remote geometry to local 

**/


void G4CXOpticks::saveGeometry() const
{
    // SGeo::DefaultDir() was giving null : due to static const depending on static const
    const char* dir = SEventConfig::OutFold() ;  
    LOG(LEVEL)  << "dir [" << ( dir ? dir : "-" )  ; 
    saveGeometry(dir) ; 
}



void G4CXOpticks::saveGeometry(const char* dir_) const
{
    LOG(LEVEL) << " dir_ " << dir_ ; 
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    LOG(LEVEL) << "[ " << ( dir ? dir : "-" ) ; 
    LOG(info)  << "[ " << ( dir ? dir : "-" ) ; 
    std::cout << "G4CXOpticks::saveGeometry [ " << ( dir ? dir : "-" ) << std::endl ;

    if(wd) U4GDML::Write(wd, dir, "origin.gdml" );  // world 
    if(fd) fd->save(dir) ; 
    //if(cx) cx->save(dir);  // debug serialization of OptixInstance that was never used 

    LOG(LEVEL) << "] " << ( dir ? dir : "-" ) ; 
}




/**
G4CXOpticks::SaveGeometry
----------------------------

Saving geometry with this method requires BOTH:

1. invoking this method from your Opticks integration code, call must be after SetGeometry 
2. have defined the below envvar configuring the directory to save into, conventionally::

   export G4CXOpticks__SaveGeometry_DIR=$HOME/.opticks/GEOM/$GEOM  

NB this is distinct from saving when setGeometry is run as
that is too soon to save when additional SSim information 
is added.

HMM : THIS MAY NO LONGER BE TRUE AS NOW USING  SSim::CreateOrReuse
SO CAN NOW SSim::Add... BEFORE G4CXOpticks::SetGeometry

IF THE setGeometry_saveGeomery proves sufficient can remove this 

**/

void G4CXOpticks::SaveGeometry() // static
{
    LOG_IF(error, INSTANCE==nullptr) << " INSTANCE nullptr : NOTHING TO SAVE " ; 
    if(INSTANCE == nullptr) return ; 
  
    const char* dir = ssys::getenvvar(SaveGeometry_KEY) ;
    LOG_IF(LEVEL, dir == nullptr ) << " not saving as envvar not set " << SaveGeometry_KEY  ;  
    if(dir == nullptr) return ; 
    LOG(info) << " save to dir " << dir << " configured via envvar " << SaveGeometry_KEY ; 
    INSTANCE->saveGeometry(dir); 
}


