/**
G4CXOpticks.cc
================

Q: Why is QSim exposed here, can that be hidden inside cx(CSGOptiX) ?
A: WIP: moving the QSim methods to be called via CSGOptiX

**/



#include <csignal>
#include "SLOG.hh"


#include "spath.h"
#include "ssys.h"

#include "SEvt.hh"
#include "SSim.hh"
#include "SOpticksResource.hh"

#include "U4VolumeMaker.hh"
#include "U4Recorder.hh"

#include "SEventConfig.hh"
#include "U4GDML.h"
#include "U4Tree.h"

#include "CSGFoundry.h"

#include "CSGOptiX.h"


#ifdef WITH_QS
#include "QSim.hh"
#endif

#include "G4CXOpticks.hh"


const plog::Severity G4CXOpticks::LEVEL = SLOG::EnvLevel("G4CXOpticks", "DEBUG");

U4SensorIdentifier* G4CXOpticks::SensorIdentifier = nullptr ;
void G4CXOpticks::SetSensorIdentifier( U4SensorIdentifier* sid ){ SensorIdentifier = sid ; }  // static


G4CXOpticks* G4CXOpticks::INSTANCE = nullptr ;
G4CXOpticks* G4CXOpticks::Get(){ return INSTANCE ; }
const U4Tree* G4CXOpticks::GetU4Tree(){ return INSTANCE ? INSTANCE->tr : nullptr ; }


/**
G4CXOpticks::SetGeometry
--------------------------

Called for example from Detector framework LSExpDetectorConstruction_Opticks::Setup

**/

G4CXOpticks* G4CXOpticks::SetGeometry()
{
    G4CXOpticks* gx = new G4CXOpticks ;
    gx->setGeometry();
    return gx ;
}

G4CXOpticks* G4CXOpticks::SetGeometryFromGDML()
{
    G4CXOpticks* gx = new G4CXOpticks ;
    gx->setGeometryFromGDML();
    return gx ;
}



G4CXOpticks* G4CXOpticks::SetGeometry(const G4VPhysicalVolume* world)
{
    G4CXOpticks* gx = new G4CXOpticks ;
    gx->setGeometry(world);
    return gx ;
}



/**
G4CXOpticks::SetGeometry_JUNO
------------------------------

Invoked from JUNOSW LSExpDetectorConstruction_Opticks::Setup

* Moved high level JUNO setup from there to here for faster update cycle.

**/


G4CXOpticks* G4CXOpticks::SetGeometry_JUNO(const G4VPhysicalVolume* world, const G4VSensitiveDetector* sd, NPFold* jpmt, const NP* jlut )
{
    LOG(LEVEL) << "[" ;

    // boot SSim adding extra juno PMT info
    SSim::CreateOrReuse();   // done previously by G4CXOpticks::G4CXOpticks in opticksMode > 0 or here in opticksMode:0
    SSim::AddExtraSubfold("jpmt", jpmt );
    SSim::AddMultiFilm(snam::MULTIFILM, jlut);

    SEvt::CreateOrReuse() ;  // creates 1/2 SEvt depending on OPTICKS_INTEGRATION_MODE (which via above assert matches opticksMode)


    int opticksMode = SEventConfig::IntegrationMode();
    if(opticksMode == 0 || opticksMode == 2) SetNoGPU() ;

    LOG(info) << "[ WITH_G4CXOPTICKS opticksMode " << opticksMode << " sd " << sd  ;

    G4CXOpticks* gx = nullptr ;

    if( opticksMode == 0 || opticksMode == 1 || opticksMode == 3 || opticksMode == 2 )
    {
        gx = SetGeometry(world) ;
        SaveGeometry();
    }

    LOG(LEVEL) << "]" ;
    return gx ;
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
#ifdef WITH_QS
    qs(nullptr),
#endif
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
#ifdef WITH_QS
       << " qs " << ( qs ? "Y" : "N" )
#endif
       ;
    std::string s = ss.str();
    return s ;
}

/**
G4CXOpticks::setGeometryFromGDML
-----------------------------------

**/

void G4CXOpticks::setGeometryFromGDML()
{
    LOG(LEVEL) << " argumentless " ;

    if(spath::has_CFBaseFromGEOM())
    {
        LOG(LEVEL) << " has_CFBaseFromGEOM " ;
        setGeometry(spath::Resolve("$CFBaseFromGEOM/origin.gdml"));
    }
    else
    {
        LOG(fatal) << " failed to setGeometryFromGDML " ;
        assert(0);
    }
}


void G4CXOpticks::setGeometry()
{
    if(spath::has_CFBaseFromGEOM())
    {
        LOG(LEVEL) << "[ CSGFoundry::Load " ;
        CSGFoundry* cf = CSGFoundry::Load() ;
        LOG(LEVEL) << "] CSGFoundry::Load " ;

        LOG(LEVEL) << "[ setGeometry(cf)  " ;
        setGeometry(cf);
        LOG(LEVEL) << "] setGeometry(cf)  " ;
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

Geometry translation from Geant4 to NVIDIA OptiX acceleration structures with
associated shader binding table (SBT) data and CUDA programs
is done in several stages::


          u4/U4Tree                  CSG/CSGImport                  CSGOptiX
      G4 —--———————> sysrap/stree.h —————-----——> CSG/CSGFoundry ———-----——> OptiX AS

First stage is steered by U4Tree::Create which traverses the Geant4 tree of
volumes collecting geometry information into the sysrap/stree.h data structure.
Note that the stree.h comes from the low level sysrap package which does not
depend on Geant4.  The stree is persisted into folders of .txt lists
and .npy arrays located at::

    $HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree

Where the "$GEOM" variable is a user selected geometry identifer such as::

     J23_1_0_rc3_ok0


* U4Tree/stree+SSim replaces the former GGeo+X4+.. packages

**/


void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
{
    LOG(LEVEL) << "[ G4VPhysicalVolume world " << world ;
    assert(world);
    wd = world ;

    assert(sim && "sim instance should have been grabbed/created in ctor" );
    stree* st = sim->get_tree();

    LOG(LEVEL) << "[U4Tree::Create " ;
    tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    LOG(LEVEL) << "]U4Tree::Create " ;


    LOG(LEVEL) << "[SSim::initSceneFromTree" ;
    sim->initSceneFromTree(); // not so easy to do at lower level as do not want to change to SSim arg to U4Tree::Create for headeronly testing
    LOG(LEVEL) << "]SSim::initSceneFromTree" ;


    LOG(LEVEL) << "[CSGFoundry::CreateFromSim" ;
    CSGFoundry* fd_ = CSGFoundry::CreateFromSim() ; // adopts SSim::INSTANCE
    LOG(LEVEL) << "]CSGFoundry::CreateFromSim" ;


    LOG(LEVEL) << "[setGeometry(fd_)" ;
    setGeometry(fd_);
    LOG(LEVEL) << "]setGeometry(fd_)" ;

    LOG(info) << Desc() ;

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

    bool hasDevice = SEventConfig::HasDevice();

    if(NoGPU == false && hasDevice == true)
    {
        LOG(LEVEL) << "[ CSGOptiX::Create " ;
        cx = CSGOptiX::Create(fd);   // uploads geometry to GPU
        LOG(LEVEL) << "] CSGOptiX::Create " ;
    }
    else
    {
        LOG(info)
            << " skip CSGOptiX::Create as NoGPU set OR failed to detect CUDA capable device "
            << " NoGPU " << NoGPU
            << " hasDevice " << ( hasDevice ? "YES" : "NO " )
            ;
    }

#ifdef WITH_QS
    qs = cx ? cx->sim : nullptr ;   // QSim

    QSim* qsg = QSim::Get()  ;

    LOG(LEVEL)
        << " cx " << ( cx ? "Y" : "N" )
        << " qs " << ( qs ? "Y" : "N" )
        << " QSim::Get " << ( qsg ? "Y" : "N" )
        ;

    assert( qs == qsg );
#endif


    LOG(LEVEL) << "] fd " << fd ;


    // try moving this later, so can save things from cx and qs for debug purposes
    if( setGeometry_saveGeometry != nullptr )
    {
        LOG(info) << "[ G4CXOpticks__setGeometry_saveGeometry " ;
        saveGeometry(setGeometry_saveGeometry);
        LOG(info) << "] G4CXOpticks__setGeometry_saveGeometry " ;
    }
}




std::string G4CXOpticks::descSimulate() const
{
    SEvt* sev = SEvt::Get_EGPU() ;
    assert(sev);
    std::stringstream ss ;
    ss
       << "[G4CXOpticks::descSimulate" << std::endl
       << sev->descSimulate()
       << "]G4CXOpticks::descSimulate" << std::endl
       ;

    std::string str = ss.str();
    return str ;
}

/**
G4CXOpticks::simulate
-----------------------

GPU launch doing generation and simulation done here

**/


void G4CXOpticks::simulate(int eventID, bool reset_ )
{
    LOG_IF(fatal, NoGPU) << "NoGPU SKIP" ;
    if(NoGPU) return ;

    LOG(LEVEL) << "[ "  << eventID;
    LOG(LEVEL) << desc() ;

    assert(cx);

#ifdef WITH_QS
    assert(qs);
    qs->simulate(eventID, reset_ );
#else
    cx->simulate(eventID, reset_);
#endif


    LOG(LEVEL) << "] " << eventID ;

}

/**
G4CXOpticks::reset
---------------------

This needs to be called after invoking G4CXOpticks::simulate
when argument reset:false has been used in order to allow copy hits
from the opticks/SEvt into other collections prior to invoking
the reset.

**/

void G4CXOpticks::reset(int eventID)
{
    LOG_IF(fatal, NoGPU) << "NoGPU SKIP" ;
    if(NoGPU) return ;

    assert( SEventConfig::IsRGModeSimulate() );

    unsigned num_hit_0 = SEvt::GetNumHit_EGPU() ;
    LOG(LEVEL) << "[ " << eventID << " num_hit_0 " << num_hit_0  ;

#ifdef WITH_QS
    assert(qs);
    qs->reset(eventID);
#else
    cx->reset(eventID);
#endif

    unsigned num_hit_1 = SEvt::GetNumHit_EGPU() ;
    LOG(LEVEL) << "] " << eventID << " num_hit_1 " << num_hit_1  ;
}





void G4CXOpticks::simtrace(int eventID)
{
    LOG_IF(fatal, NoGPU) << "NoGPU SKIP" ;
    if(NoGPU) return ;

    LOG(LEVEL) << "[" ;

#ifdef WITH_QS
    assert(qs);
    qs->simtrace(eventID);
#else
    assert(cx);
    cx->simtrace(eventID);
#endif


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

What is saved includes the gdml and CSGFoundry folder.
Grab that locally::

    GEOM get  ## grab the remote geometry to local

**/


void G4CXOpticks::saveGeometry() const
{
    const char* dir = SEventConfig::OutFold() ;
    LOG(LEVEL)  << "dir [" << ( dir ? dir : "-" )  ;
    saveGeometry(dir) ;
}



void G4CXOpticks::saveGeometry(const char* dir_) const
{
    LOG(LEVEL) << " dir_ " << dir_ ;
    const char* dir = spath::Resolve(dir_);
    LOG(LEVEL) << "[ " << ( dir ? dir : "-" ) ;
    LOG(info)  << "[ " << ( dir ? dir : "-" ) ;
    std::cout << "G4CXOpticks::saveGeometry [ " << ( dir ? dir : "-" ) << std::endl ;

    if(wd) U4GDML::Write(wd, dir, "origin.gdml" );  // world
    if(fd) fd->save(dir) ;

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



/**
G4CXOpticks::SensitiveDetector_Initialize
-------------------------------------------

Optional lifecycle method intended to be called from Geant4 integration
G4VSensitiveDetector::Initialize

**/

void G4CXOpticks::SensitiveDetector_Initialize(int eventID)
{
    LOG(LEVEL) << " eventID " << eventID ;
    U4Recorder* recorder = U4Recorder::Get();
    if(recorder)
    {
        recorder->BeginOfEventAction_(eventID);
    }
}


/**
G4CXOpticks::SensitiveDetector_EndOfEvent
-------------------------------------------

Optional lifecycle method intended to be called from Geant4 integration
G4VSensitiveDetector::EndOfEvent

The Geant4 call order gleaned from g4-cls G4EventManager
with the related calls to G4CXOpticks and U4Recorder
that were used previously::

    G4VSensitiveDetector::Initialize(G4HCofThisEvent*HCE)
    (eg junoSD_PMT_v2::Initialize)

    G4UserEventAction::BeginOfEventAction  ==> U4Recorder::BeginOfEventAction


    G4VSensitiveDetector::EndOfEvent(G4HCofThisEvent*HCE)   ==> G4CXOpticks::simulate : AS HITS NEEDED HERE
    (eg junoSD_PMT_v2::EndOfEvent)

    G4UserEventAction::EndOfEventAction    ==> U4Recorder::EndOfEventAction




The above old call order is problematic as the simulate call needs the gensteps
from the SEvt::ECPU that is only wrapped up at U4Recorder::EndOfEventAction.





This SensitiveDetector pair of methods enables rearranging the order::

    G4VSensitiveDetector::Initialize       ==> U4Recorder::BeginOfEventAction
    (eg junoSD_PMT_v2::Initialize)

    G4UserEventAction::BeginOfEventAction


    G4VSensitiveDetector::EndOfEvent       ==> U4Recorder::EndOfEventAction,
    (eg junoSD_PMT_v2::EndOfEvent)
                                           ==> G4CXOpticks::simulate : AS HITS NEEDED HERE

    G4UserEventAction::EndOfEventAction

Note that the Stepping and Tracking actions handled by
the U4Recorder can proceed unchanged by this.

**/

void G4CXOpticks::SensitiveDetector_EndOfEvent(int eventID)
{
    LOG(LEVEL) << " eventID " << eventID ;
    U4Recorder* recorder = U4Recorder::Get();
    if(recorder)
    {
        recorder->EndOfEventAction_(eventID);
    }
}


