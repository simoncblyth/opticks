/**
G4CXApp.h  : Geant4 Application integrated with G4CXOpticks within a single header 
=========================================================================================

This is based upon u4/U4App.h with the addition of G4CXOpticks 

Note that the methods are not inlined, but that does not matter as this should only be included
once into the main.  

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

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "ssys.h"
#include "SEventConfig.hh"
#include "NP.hh"
#include "sframe.h"

#include "U4Material.hh"
#include "U4VolumeMaker.hh"
#include "U4Recorder.hh"
#include "U4Random.hh"
#include "U4Physics.hh"
#include "U4VPrimaryGenerator.h"

#include "G4CXOpticks.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif


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
    static char PrimaryMode(); 
    static G4ParticleGun* InitGun(); 

    G4RunManager*         fRunMgr ;  
    char                  fPrimaryMode ;  
    U4Recorder*           fRecorder ; 
    G4ParticleGun*        fGun ;  
    G4VPhysicalVolume*    fPV ; 
    G4CXOpticks*          fGX ;  

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
    virtual ~G4CXApp(); 

    static void SaveMeta(const char* savedir); 
    static G4RunManager* InitRunManager(); 
    static G4CXApp*        Create(); 
    void                 BeamOn() ;

};

const plog::Severity G4CXApp::LEVEL = info ;   // PLOG logging level control doesnt work in the main 

std::string G4CXApp::Desc()
{
    std::string phy = U4Physics::Desc() ; 
    std::string rec = U4Recorder::Desc() ; 
    std::stringstream ss ; 
    if(!phy.empty()) ss << phy  ; 
    if(!rec.empty()) ss << "/" << rec ; 
    std::string s = ss.str(); 
    return s ; 
}

char G4CXApp::PrimaryMode()
{
    char mode = '?' ; 
    const char* mode_ = ssys::getenvvar("G4CXApp__PRIMARY_MODE", "torch" ); 
    if(strcmp(mode_, "gun")   == 0) mode = 'G' ; 
    if(strcmp(mode_, "torch") == 0) mode = 'T' ; 
    if(strcmp(mode_, "iphoton") == 0) mode = 'I' ;   // CAUTION: torch and iphoton both call U4VPrimaryGenerator::GeneratePrimaries
    return mode ;   
}

G4ParticleGun* G4CXApp::InitGun()
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

G4CXApp::G4CXApp(G4RunManager* runMgr)
    :
    fRunMgr(runMgr),
    fPrimaryMode(PrimaryMode()),
    fRecorder(new U4Recorder),
    fGun(fPrimaryMode == 'G' ? InitGun() : nullptr),
    fPV(nullptr)
{
    fRunMgr->SetUserInitialization((G4VUserDetectorConstruction*)this);
    fRunMgr->SetUserAction((G4VUserPrimaryGeneratorAction*)this);
    fRunMgr->SetUserAction((G4UserRunAction*)this);
    fRunMgr->SetUserAction((G4UserEventAction*)this);
    fRunMgr->SetUserAction((G4UserTrackingAction*)this);
    fRunMgr->SetUserAction((G4UserSteppingAction*)this);
    fRunMgr->Initialize(); 

}

G4VPhysicalVolume* G4CXApp::Construct()
{ 
    LOG(info) << "[" ; 
    const G4VPhysicalVolume* pv_ = U4VolumeMaker::PV() ;
    LOG_IF(fatal, pv_ == nullptr) << " FAILED TO CREATE PV : CHECK GEOM envvar " ;  
    if(pv_ == nullptr) std::raise(SIGINT) ; 

    G4VPhysicalVolume* pv = const_cast<G4VPhysicalVolume*>(pv_);  
    fPV = pv ; 
    LOG(LEVEL) << " fPV " << ( fPV ? fPV->GetName() : "ERR-NO-PV" ) ; 

    LOG(info) << "]" ; 
    return pv ; 
}  

// pass along the message to the recorder
void G4CXApp::BeginOfRunAction(const G4Run* run)
{ 
    LOG(info) ; 
    fRecorder->BeginOfRunAction(run);   
}
void G4CXApp::EndOfRunAction(const G4Run* run)
{   
    LOG(info); 
    fRecorder->EndOfRunAction(run);     
}


/**
G4CXApp::GeneratePrimaries
------------------------------------

Other that for gun running this uses U4VPrimaryGenerator::GeneratePrimaries
which is based on SGenerate::GeneratePhotons

**/

void G4CXApp::GeneratePrimaries(G4Event* event)
{   
    LOG(LEVEL) << "[ fPrimaryMode " << fPrimaryMode  ; 

    if(fPrimaryMode == 'T')
    {
        SEvt::AddTorchGenstep();  
    }

    switch(fPrimaryMode)
    {
        case 'G': fGun->GeneratePrimaryVertex(event)              ; break ; 
        case 'T': U4VPrimaryGenerator::GeneratePrimaries(event);  ; break ; // eg from collected torch gensteps 
        case 'I': U4VPrimaryGenerator::GeneratePrimaries(event);  ; break ; // NOTE torch and iphoton are doing the same thing : misleading 
        default:  assert(0) ; break ; 
    }
    LOG(LEVEL) << "]" ; 
}

void G4CXApp::BeginOfEventAction(const G4Event* event)
{  
    // TOO LATE TO SEvt::AddTorchGenstep here as GeneratePrimaries already run 
    fRecorder->BeginOfEventAction(event); 
}
void G4CXApp::EndOfEventAction(const G4Event* event)
{   
    fRecorder->EndOfEventAction(event);  

    const char* savedir = SEvt::GetSaveDir(1); 
    SaveMeta(savedir); 

#if defined(WITH_PMTSIM) && defined(POM_DEBUG)
    PMTSim::ModelTrigger_Debug_Save(savedir) ; 
#else
    LOG(info) << "not-(WITH_PMTSIM and POM_DEBUG)"  ; 
#endif
}

void G4CXApp::PreUserTrackingAction(const G4Track* trk){  fRecorder->PreUserTrackingAction(trk); }
void G4CXApp::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk); }
void G4CXApp::UserSteppingAction(const G4Step* step){     fRecorder->UserSteppingAction(step) ; }


G4CXApp::~G4CXApp(){  G4GeometryManager::GetInstance()->OpenGeometry(); }
// G4GeometryManager::OpenGeometry is needed to avoid cleanup warning


void G4CXApp::SaveMeta(const char* savedir) // static
{
    if(savedir == nullptr)
    {
        LOG(error) << " NULL savedir " ; 
        return ; 
    }  
    // U4Recorder::SaveMeta(savedir);   // try moving to U4Recorder::EndOfEventAction
    U4VolumeMaker::SaveTransforms(savedir) ;   
}

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
    fRunMgr->BeamOn(ssys::getenvint("BeamOn",1)); 
}


