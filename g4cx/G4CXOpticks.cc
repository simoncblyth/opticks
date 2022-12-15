
#include "SLOG.hh"

#include "scuda.h"
#include "sqat4.h"
#include "sframe.h"

#include "SSys.hh"
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
#include "CSG_GGeo_Convert.h"
#include "CSGOptiX.h"
#include "QSim.hh"

#include "G4CXOpticks.hh"



// OLD WORLD HEADERS STILL NEEDED UNTIL REJIG TRANSLATION 
#include "Opticks.hh"
#include "X4Geo.hh"
#include "GGeo.hh"



const plog::Severity G4CXOpticks::LEVEL = SLOG::EnvLevel("G4CXOpticks", "DEBUG"); 

const U4SensorIdentifier* G4CXOpticks::SensorIdentifier = nullptr ; 
void G4CXOpticks::SetSensorIdentifier( const U4SensorIdentifier* sid ){ SensorIdentifier = sid ; }  // static 


G4CXOpticks* G4CXOpticks::INSTANCE = nullptr ; 
G4CXOpticks* G4CXOpticks::Get(){ return INSTANCE ; } 
// formerly instanciated when INSTANCE nullptr, but its better to require more care with when the instanciation is done

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
    LOG(LEVEL) << "placeholder mimic G4Opticks " ; 
}


G4CXOpticks::G4CXOpticks()
    :
    sim(SSim::Create()),      // HMM: is there a better place to do this frm lower level ?
    tr(nullptr),
    wd(nullptr),
    gg(nullptr),
    fd(nullptr), 
    cx(nullptr),
    qs(nullptr),
    t0(schrono::stamp())
{
    init();
}

void G4CXOpticks::init()
{
    // SEventConfig::Initialize();     
    // config depends on SEventConfig::SetEventMode OR OPTICKS_EVENTMODE envvar 
    // DONE: moved SEventConfig::Initialize to SEvt::SEvt for better "locus", hence avoiding repetition 

    INSTANCE = this ; 
    LOG(LEVEL) << Desc() << std::endl << desc(); 
}

G4CXOpticks::~G4CXOpticks()
{
    schrono::TP t1 = schrono::stamp(); 
    double dt = schrono::duration(t0, t1 );
    //LOG(LEVEL) << "lifetime (s) " << std::scientific << dt << " s " ; 
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
       << " sim " << sim 
       << " tr " << tr 
       << " wd " << wd 
       << " gg " << gg
       << " fd " << fd
       << " cx " << cx
       << " qs " << qs
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

    if(SSys::hasenvvar(SOpticksResource::OpticksGDMLPath_))
    {
        LOG(LEVEL) << " OpticksGDMLPath " ; 
        setGeometry(SOpticksResource::OpticksGDMLPath()); 
    }
    else if(SSys::hasenvvar(SOpticksResource::SomeGDMLPath_))
    {
        LOG(LEVEL) << " SomeGDMLPath " ; 
        setGeometry(SOpticksResource::SomeGDMLPath()); 
    }
    else if(SSys::hasenvvar(SOpticksResource::CFBASE_))
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
        LOG(LEVEL) << "[ GDMLPathFromGEOM " ; 
        setGeometry(SOpticksResource::GDMLPathFromGEOM()) ; 
        LOG(LEVEL) << "] GDMLPathFromGEOM " ; 
    }
    else if(SSys::hasenvvar("GEOM"))
    {
        LOG(LEVEL) << " GEOM/U4VolumeMaker::PV " ; 
        setGeometry( U4VolumeMaker::PV() );  // this may load GDML using U4VolumeMaker::PVG if "GEOM"_GDMLPath is defined   
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

U4Tree/stree+SSim aspiring to replace the GGeo+X4+.. packages 

HMM: need a way to distingish between a re-animated world coming via GDML save/load  
and a live original world : as need to do things a bit differently in each case.

* could determine this by noticing the lack of SensitiveDetectors in GDML re-animated world,
  (but thats kinda heavy way to determine one bit) OR by passing a signal along with the 
  world to show that it has been re-animated  


Q: Is stree.h/st in actual use yet ? Where ? What parts of GGeo does that replace ?
A: Already playing vital role as *tree* member of CSG_GGeo/CSG_GGeo_Convert.cc 

**/


void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
{
    LOG(LEVEL) << " G4VPhysicalVolume world " << world ; 
    assert(world); 
    wd = world ;

    sim = SSim::Create();  
    stree* st = sim->get_tree(); 
    // TODO: sim argument, not st : or do SSim::Create inside U4Tree::Create 
    tr = U4Tree::Create(st, world, SensorIdentifier ) ;

  
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
}

/**
G4CXOpticks::setGeometry
---------------------------

Prior to CSGOptiX::Create the SEvt instance is created. 

Q: is there a more general place for SEvt hookup ?
A: SSim could hold the SEvt together with stree ?

But SEvt feels like it should be separate, 
as the SSim focus is initialization and SEvt focis is post-init.  

**/


const char* G4CXOpticks::setGeometry_saveGeometry = SSys::getenvvar("G4CXOpticks__setGeometry_saveGeometry") ;

void G4CXOpticks::setGeometry(CSGFoundry* fd_)
{
    fd = fd_ ; 

    if( setGeometry_saveGeometry )
    {
        LOG(LEVEL) << "[ G4CXOpticks__setGeometry_saveGeometry " ;  
        saveGeometry(setGeometry_saveGeometry); 
        LOG(LEVEL) << "] G4CXOpticks__setGeometry_saveGeometry " ;  
    }

#ifdef __APPLE__
    LOG(fatal) << " __APPLE__ early exit " ; 
    return ; 
#endif
    LOG(LEVEL) << "[ fd " << fd ; 

    LOG(LEVEL) << " [ new SEvt " ; 
    SEvt* sev = new SEvt ; 
    LOG(LEVEL) << " ] new SEvt " ; 
    
    sev->setReldir("ALL"); 
    sev->setGeo((SGeo*)fd);   

    LOG(LEVEL) << "[ CSGOptiX::Create " ;  
    cx = CSGOptiX::Create(fd);   // uploads geometry to GPU 
    LOG(LEVEL) << "] CSGOptiX::Create " ;  
    qs = cx->sim ; 
    LOG(LEVEL)  << " cx " << cx << " qs " << qs << " QSim::Get " << QSim::Get() ; 


    LOG(LEVEL) << "] fd " << fd ; 
}


/**
G4CXOpticks::simulate
------------------------

To enable saving of SEvt for low level NumPy debugging, define the envvar::

   export G4CXOpticks__simulate_saveEvent=1

Note that the SEvt component arrays will overrite themselves
if the SEvt index is not incremented with "SEvt::SetIndex" 
for each call to G4CXOpticks::simulate. 


HMM: note that all of G4CXOpticks::simulate could be down in an SSim::simulate, 
      just needs protocol for QSim::simulate call 
TODO: compare with B side to see if that makes sense when viewed from A and B directions 


**/

//const bool G4CXOpticks::simulate_saveEvent = true ;
const bool G4CXOpticks::simulate_saveEvent = SSys::getenvbool("G4CXOpticks__simulate_saveEvent") ;

void G4CXOpticks::simulate()
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif
    LOG(LEVEL) << "[" ; 
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
    //unsigned num_photon  = sev->getNumPhotonFromGenstep(); 
    unsigned num_photon  = sev->getNumPhotonCollected(); 

    LOG(LEVEL) 
        << "[ num_genstep " << num_genstep
        << " num_photon " << num_photon
        << " has_input_photon " << has_input_photon
        ;

    //]

    qs->simulate();   // GPU launch doing generation and simulation here 

    sev->gather();   // downloads components configured by SEventConfig::CompMask 

    unsigned num_hit = sev->getNumHit() ; 
    bool is_undef = num_hit == SEvt::UNDEF ; 


    int sev_index = SEvt::GetIndex() ;

    LOG(LEVEL) 
       << "] num_hit " << num_hit 
       << " is_undef " << is_undef 
       << " sev_index " << sev_index 
       << " simulate_saveEvent " << simulate_saveEvent
       ;

    LOG(LEVEL) << " sev.descFull " << std::endl << sev->descFull() ; 

    if(simulate_saveEvent)
    {
        LOG(LEVEL) << "[ G4CXOpticks__simulate_saveEvent" << sev_index ; 
        saveEvent(); 
        LOG(LEVEL) << "] G4CXOpticks__simulate_saveEvent" << sev_index ; 
    }

    LOG(LEVEL) << "]" ; 
}

void G4CXOpticks::simtrace()
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif
    LOG(LEVEL) << "[" ; 
    assert(cx); 
    assert(qs); 
    assert( SEventConfig::IsRGModeSimtrace() ); 
                           
    SEvt* sev = SEvt::Get();  assert(sev); 

    sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    sev->setFrame(fr);   

    cx->setFrame(fr);    
    // Q: why does cx need the frame ?
    // A: the rendering viewpoint or simtrace grid is based on the frame center, extent and transforms 

    qs->simtrace(); 
    LOG(LEVEL) << "]" ; 
}

void G4CXOpticks::render()
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif
    LOG(LEVEL) << "[" ; 
    assert( cx ); 
    assert( SEventConfig::IsRGModeRender() ); 
    cx->render_snap() ; 
    LOG(LEVEL) << "]" ; 
}

void G4CXOpticks::saveEvent() const 
{
#ifdef __APPLE__
     LOG(fatal) << " APPLE skip " ; 
     return ; 
#endif
    LOG(LEVEL) << "[" ; 
    SEvt* sev = SEvt::Get(); 
    if(sev == nullptr) return ; 

    LOG(LEVEL) << "[ sev.save " ; 
    sev->save(); 
    LOG(LEVEL) << "] sev.save " ; 

    /*
    if( LEVEL == info && SEventConfig::IsRGModeSimulate() )
    { 
        LOG(LEVEL) << sev->descPhoton() ; 
        LOG(LEVEL) << sev->descLocalPhoton() ; 
        LOG(LEVEL) << sev->descFramePhoton() ; 
    }
    */
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

    getgeom () 
    { 
        GEOM=${GEOM:-ntds3};
        BASE=/tmp/$USER/opticks/GEOM/$GEOM/G4CXOpticks;
        source $OPTICKS_HOME/bin/rsync.sh $BASE
    }

**/


void G4CXOpticks::saveGeometry() const
{
    const char* dir = SEventConfig::OutFold() ;  // SGeo::DefaultDir() was giving null : due to static const depending on static const
    LOG(LEVEL)  << "dir [" << ( dir ? dir : "-" )  ; 
    saveGeometry(dir) ; 
}
void G4CXOpticks::saveGeometry(const char* dir_) const
{
    const char* dir = SPath::Resolve(dir_, DIRPATH); 
    LOG(LEVEL) << "[ " << ( dir ? dir : "-" ) ; 

    if(fd) fd->save(dir) ; 
    if(gg) gg->save_to_dir(dir) ; 
    if(wd) U4GDML::Write(wd, dir, "origin.gdml" );

    LOG(LEVEL) << "] " << ( dir ? dir : "-" ) ; 
}

