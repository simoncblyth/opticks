/**
G4CXApp.h  : Geant4 Application integrated with G4CXOpticks within a single header 
=========================================================================================

This ~300 line header does everything to setup a Geant4 app and integrate G4CXOpticks
plus a U4Recorder instance that collects the details of the Geant4 simulation in Opticks SEvt 
format to facilitate comparison of Opticks and Geant4 optical simulations. 

Note that the methods are not inlined, but that does not matter as this should only be included
once into the main. This was initially based upon u4/U4App.h with the addition of G4CXOpticks 

Geometry setup in G4CXApp::Construct is done by U4VolumeMaker::PV which is controlled by the GEOM envvar.  

**/

#include <csignal>

#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4UserSteppingAction.hh"

#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleGun.hh"
#include "G4GeometryManager.hh"


#include "ssys.h"
#include "sframe.h"

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SSim.hh"
#include "SEventConfig.hh"
#include "SRM.h"
#include "SGenerate.h"
#include "SEvent.hh"


#include "U4Material.hh"
#include "U4VolumeMaker.hh"
#include "U4Recorder.hh"
#include "U4Random.hh"
#include "U4Physics.hh"
#include "U4SensitiveDetector.hh"
#include "U4VPrimaryGenerator.h"

#include "G4CXOpticks.hh"


struct G4CXApp
    : 
    public G4UserRunAction,  
    public G4UserEventAction,
    public G4UserTrackingAction,
    public G4UserSteppingAction,
    public G4VUserPrimaryGeneratorAction,
    public G4VUserDetectorConstruction
{
    static const plog::Severity LEVEL ; 
    static std::string Desc(); 
    static G4ParticleGun* InitGun(); 
    static U4SensitiveDetector* InitSensDet(); 

    G4RunManager*         fRunMgr ;  
    U4Recorder*           fRecorder ; 
    G4ParticleGun*        fGun ;  
    U4SensitiveDetector*  fSensDet ; 
    G4VPhysicalVolume*    fPV ; 
    

    G4VPhysicalVolume* Construct(); 

    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void GeneratePrimaries(G4Event* evt); 
    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PreUserTrackingAction(const G4Track*);
    void PostUserTrackingAction(const G4Track*);

    void UserSteppingAction(const G4Step*);


    G4CXApp(G4RunManager* runMgr); 
    static void OpenGeometry() ; 
    virtual ~G4CXApp(); 

    static G4RunManager* InitRunManager(); 
    static G4CXApp*        Create(); 
    void                 BeamOn() ;
    static void          Main(); 

};

const plog::Severity G4CXApp::LEVEL = info ;   // PLOG logging level control doesnt work in the main 

std::string G4CXApp::Desc() // static
{
    std::string phy = U4Physics::Desc() ; 
    std::string rec = U4Recorder::Desc() ; 
    std::stringstream ss ; 
    if(!phy.empty()) ss << phy  ; 
    if(!rec.empty()) ss << "/" << rec ; 
    std::string s = ss.str(); 
    return s ; 
}

G4ParticleGun* G4CXApp::InitGun() // static
{
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particle = particleTable->FindParticle("e+");
    LOG(LEVEL) << " particle " << particle ; 
    G4ParticleGun* gun = new G4ParticleGun(1) ;   
    gun->SetParticleDefinition(particle);
    gun->SetParticleTime(0.0*CLHEP::ns);
    gun->SetParticlePosition(G4ThreeVector(0.0*CLHEP::cm,0.0*CLHEP::cm,0.0*CLHEP::cm));
    gun->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
    gun->SetParticleEnergy(1.0*MeV); 
    return gun ; 
}

U4SensitiveDetector* G4CXApp::InitSensDet() // static
{
    const char* sdn = ssys::getenvvar("G4CXApp__SensDet", "PMTSDMgr" ) ; 
    U4SensitiveDetector* sd = sdn ? new U4SensitiveDetector(sdn) : nullptr ; 
    std::cout 
        << "G4CXApp::InitSensDet" 
        << " sdn " << ( sdn ? sdn : "-" )
        << " sd " << ( sd ? "YES" : "NO " )
        << std::endl
        << U4SensitiveDetector::Desc()
        << std::endl
        ; 
    return sd ; 
}

G4CXApp::G4CXApp(G4RunManager* runMgr)
    :
    fRunMgr(runMgr),
    fRecorder(new U4Recorder),
    fGun(SEventConfig::IsRunningModeGun() ? InitGun() : nullptr),
    fSensDet(InitSensDet()),
    fPV(nullptr)
{
    fRunMgr->SetUserInitialization((G4VUserDetectorConstruction*)this);
    fRunMgr->SetUserAction((G4VUserPrimaryGeneratorAction*)this);
    fRunMgr->SetUserAction((G4UserRunAction*)this);
    fRunMgr->SetUserAction((G4UserEventAction*)this);
    fRunMgr->SetUserAction((G4UserTrackingAction*)this);
    fRunMgr->SetUserAction((G4UserSteppingAction*)this);
    fRunMgr->Initialize(); 

    LOG(info) << std::endl << U4Recorder::Desc() ;  
}

G4VPhysicalVolume* G4CXApp::Construct()
{ 
    LOG(info) << "[" ; 
    const G4VPhysicalVolume* pv_ = U4VolumeMaker::PV() ;
    LOG_IF(fatal, pv_ == nullptr) 
        << " FAILED TO CREATE PV : CHECK GEOM envvar " 
        << std::endl
        << U4VolumeMaker::Desc()
        ;  
   
    if(pv_ == nullptr) std::raise(SIGINT) ; 

    G4VPhysicalVolume* pv = const_cast<G4VPhysicalVolume*>(pv_);  
    fPV = pv ; 
    LOG(LEVEL) << " fPV " << ( fPV ? fPV->GetName() : "ERR-NO-PV" ) ; 

    LOG(info) << "]" ; 
    
    // Collect extra JUNO PMT info only when persisted NPFold exists.
    SSim::AddExtraSubfold("jpmt", "$PMTSimParamData_BASE" ); 

    if(SEventConfig::GPU_Simulation())
    {
        G4CXOpticks::SetGeometry(pv_) ; 
        G4CXOpticks::SaveGeometry() ; 
    }
    else
    {
        LOG(LEVEL) << " SEventConfig::GPU_Simulation() false : SKIP G4CXOpticks::SetGeometry " ; 
    }

    return pv ; 
}  

void G4CXApp::BeginOfRunAction(const G4Run* run){ fRecorder->BeginOfRunAction(run);   }
void G4CXApp::EndOfRunAction(const G4Run* run){   fRecorder->EndOfRunAction(run);     }


/**
G4CXApp::GeneratePrimaries
------------------------------------

Other that for gun running this uses U4VPrimaryGenerator::GeneratePrimaries
which is based on SGenerate::GeneratePhotons

As U4VPrimaryGenerator::GeneratePrimaries needs the gensteps early 
(before U4Recorder::BeginOfEventAction invokes SEvt::beginOfEvent)
to allow Geant4 to use SGenerate::GeneratePhotons it is necessary 
to SEvt::addTorchGenstep here. 

But thats means cannot use the normal EGPU pattern of adding gensteps 
in the SEvt::beginOfEvent call.  
That causes kludgy SEvt::addFrameGenstep

**/

void G4CXApp::GeneratePrimaries(G4Event* event)
{  
    G4int eventID = event->GetEventID(); 
 
    LOG(LEVEL) << "[ SEventConfig::RunningModeLabel " << SEventConfig::RunningModeLabel() << " eventID " << eventID ; 
    SEvt* sev = SEvt::Get_ECPU();  
    assert(sev); 

    if(SEventConfig::IsRunningModeGun())
    {
        LOG(fatal) << " THIS MODE NEEDS WORK ON U4PHYSICS " ; 
        std::raise(SIGINT); 
        fGun->GeneratePrimaryVertex(event) ; 
    }
    else if(SEventConfig::IsRunningModeTorch())
    {
        const NP* gs = SEvent::MakeTorchGensteps() ;        
        NP* ph = SGenerate::GeneratePhotons(gs); 
        U4VPrimaryGenerator::GeneratePrimaries_From_Photons(event, ph);
    }
    else if(SEventConfig::IsRunningModeInputPhoton())
    {
        NP* ph = sev->getInputPhoton(); 
        U4VPrimaryGenerator::GeneratePrimaries_From_Photons(event, ph) ; 
    }
    else if(SEventConfig::IsRunningModeInputGenstep())
    {
        LOG(fatal) << "General InputGensteps with Geant4 not implemented, use eg cxs_min.sh to do that with Opticks " ; 
        std::raise(SIGINT); 
    }
    LOG(LEVEL) << "] " << " eventID " << eventID  ; 
}

/**
G4CXApp::BeginOfEventAction
----------------------------

Its too late to SEvt::AddTorchGenstep here as GeneratePrimaries already run 

**/

void G4CXApp::BeginOfEventAction(const G4Event* event){  fRecorder->BeginOfEventAction(event); }

/**
G4CXApp::EndOfEventAction
---------------------------

::

    SEvt::ECPU endOfEvent
    G4CXOpticks::simulate
          SEvt::EGPU beginOfEvent
          ... launch .. 
          SEvt::EGPU endOfEvent

* GPU begin after CPU end 

**/


void G4CXApp::EndOfEventAction(const G4Event* event)
{  
    fRecorder->EndOfEventAction(event);   // saves SEvt::ECPU   

    if(SEventConfig::GPU_Simulation())
    {
        G4CXOpticks* gx = G4CXOpticks::Get() ;
        int eventID = event->GetEventID() ;
        gx->simulate(eventID) ;
    }
    else
    {
         LOG(LEVEL) << " SEventConfig::GPU_Simulation() false : SKIP G4CXOpticks::simulate " ; 
    }
}

void G4CXApp::PreUserTrackingAction(const G4Track* trk){  fRecorder->PreUserTrackingAction(trk); }
void G4CXApp::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk); }
void G4CXApp::UserSteppingAction(const G4Step* step){     fRecorder->UserSteppingAction(step) ; }

void G4CXApp::OpenGeometry(){  G4GeometryManager::GetInstance()->OpenGeometry(); } // static
G4CXApp::~G4CXApp(){ OpenGeometry(); }
// G4GeometryManager::OpenGeometry is needed to avoid cleanup warning


G4RunManager* G4CXApp::InitRunManager()  // static
{
    G4VUserPhysicsList* phy = (G4VUserPhysicsList*)new U4Physics ; 
    G4RunManager* run = new G4RunManager ; 
    run->SetUserInitialization(phy) ; 
    return run ; 
}

/**
G4CXApp::Create
----------------

Geant4 requires G4RunManager to be instanciated prior to the Actions 

**/

G4CXApp* G4CXApp::Create()  // static 
{
    LOG(info) << U4Recorder::Switches() ; 

    G4RunManager* run = InitRunManager(); 
    G4CXApp* app = new G4CXApp(run); 
    return app ; 
}

void G4CXApp::BeamOn()
{
    LOG(info) << "[ " << SEventConfig::kNumEvent << "=" << SEventConfig::NumEvent()  ; 
    fRunMgr->BeamOn(SEventConfig::NumEvent()) ; 
    LOG(info) << "]" ; 
}

void G4CXApp::Main()  // static 
{
    G4CXApp* app = G4CXApp::Create() ;   
    app->BeamOn(); 
    delete app ;  // avoids "Attempt to delete the (physical volume/logical volume/solid/region) store while geometry closed" warnings 
}

