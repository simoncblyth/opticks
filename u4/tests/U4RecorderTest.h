/**
U4RecorderTest.h  : Geant4 Application in a header : U4App.h  might be a better name
=======================================================================================

Note that the methods are not inlined, but that
does not matter as this should only be included
once into the main.  This is effectively 
providing a Geant4 application in a single header. 

Geometry setup in U4RecorderTest::Construct is done by 
U4VolumeMaker::PV which is controlled by the GEOM envvar.  

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
#include "SSys.hh"
#include "SPath.hh"
#include "SEventConfig.hh"
#include "NP.hh"
#include "sframe.h"

#include "U4Material.hh"
#include "U4VolumeMaker.hh"
#include "U4Recorder.hh"
#include "U4Random.hh"
#include "U4Physics.hh"
#include "U4VPrimaryGenerator.h"



struct U4RecorderTest   // HMM: U4Action would be a better name 
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

    char                  fPrimaryMode ;  
    U4Recorder*           fRecorder ; 
    G4ParticleGun*        fGun ;  
    G4VPhysicalVolume*    fPV ; 

    G4VPhysicalVolume* Construct(); 

    void GeneratePrimaries(G4Event* evt); 

    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PreUserTrackingAction(const G4Track*);
    void PostUserTrackingAction(const G4Track*);

    void UserSteppingAction(const G4Step*);

    U4RecorderTest(G4RunManager* runMgr); 
    virtual ~U4RecorderTest(); 

};

const plog::Severity  U4RecorderTest::LEVEL = info ;   // PLOG logging level control doesnt work in the main 


std::string U4RecorderTest::Desc()
{
    std::string phy = U4Physics::Desc() ; 
    std::string rec = U4Recorder::Desc() ; 
    std::stringstream ss ; 
    if(!phy.empty()) ss << phy  ; 
    if(!rec.empty()) ss << "/" << rec ; 
    std::string s = ss.str(); 
    return s ; 
}

char U4RecorderTest::PrimaryMode()
{
    char mode = '?' ; 
    const char* mode_ = SSys::getenvvar("U4RecorderTest__PRIMARY_MODE", "gun" ); 
    if(strcmp(mode_, "gun")   == 0) mode = 'G' ; 
    if(strcmp(mode_, "torch") == 0) mode = 'T' ; 
    if(strcmp(mode_, "iphoton") == 0) mode = 'I' ;   // CAUTION: torch and iphoton both call U4VPrimaryGenerator::GeneratePrimaries
    return mode ;   
}

G4ParticleGun* U4RecorderTest::InitGun()
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

U4RecorderTest::U4RecorderTest(G4RunManager* runMgr)
    :
    fPrimaryMode(PrimaryMode()),
    fRecorder(new U4Recorder),
    fGun(fPrimaryMode == 'G' ? InitGun() : nullptr),
    fPV(nullptr)
{
    runMgr->SetUserInitialization((G4VUserDetectorConstruction*)this);
    runMgr->SetUserAction((G4VUserPrimaryGeneratorAction*)this);
    runMgr->SetUserAction((G4UserRunAction*)this);
    runMgr->SetUserAction((G4UserEventAction*)this);
    runMgr->SetUserAction((G4UserTrackingAction*)this);
    runMgr->SetUserAction((G4UserSteppingAction*)this);
    runMgr->Initialize(); 
}

G4VPhysicalVolume* U4RecorderTest::Construct()
{ 
    LOG(info) << "[" ; 

    const G4VPhysicalVolume* pv_ = U4VolumeMaker::PV() ;
    LOG_IF(fatal, pv_ == nullptr) << " FAILED TO CREATE PV " ;  
    assert(pv_);
    if(pv_ == nullptr) std::raise(SIGINT) ; 

    G4VPhysicalVolume* pv = const_cast<G4VPhysicalVolume*>(pv_);  // sensitive to GEOM envvar 
    fPV = pv ; 
    //G4Exception ("U4RecorderTest::Construct()", "000", FatalException, "FAILED TO CREATE PV");
    LOG(LEVEL) << " fPV " << ( fPV ? fPV->GetName() : "ERR-NO-PV" ) ; 

    LOG(info) << "]" ; 
    return pv ; 
}  

void U4RecorderTest::GeneratePrimaries(G4Event* event)
{   
    LOG(LEVEL) << "[ fPrimaryMode " << fPrimaryMode  ; 
    switch(fPrimaryMode)
    {
        case 'G': fGun->GeneratePrimaryVertex(event)              ; break ; 
        case 'T': U4VPrimaryGenerator::GeneratePrimaries(event);  ; break ; // eg from collected torch gensteps 
        case 'I': U4VPrimaryGenerator::GeneratePrimaries(event);  ; break ; // NOTE torch and iphoton are doing the same thing : misleading 
        default:  assert(0) ; break ; 
    }
    LOG(LEVEL) << "]" ; 
}

// pass along the message to the recorder
void U4RecorderTest::BeginOfRunAction(const G4Run* run){         fRecorder->BeginOfRunAction(run);   }
void U4RecorderTest::EndOfRunAction(const G4Run* run){           fRecorder->EndOfRunAction(run);     }
void U4RecorderTest::BeginOfEventAction(const G4Event* evt){     fRecorder->BeginOfEventAction(evt); }
void U4RecorderTest::EndOfEventAction(const G4Event* evt){       fRecorder->EndOfEventAction(evt);   }
void U4RecorderTest::PreUserTrackingAction(const G4Track* trk){  fRecorder->PreUserTrackingAction(trk); }
void U4RecorderTest::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk); }


#ifdef WITH_PMTSIM
#include "G4OpBoundaryProcess.hh"
#include "CustomG4OpBoundaryProcess.hh"
void U4RecorderTest::UserSteppingAction(const G4Step* step){     fRecorder->UserSteppingAction<CustomG4OpBoundaryProcess>(step); }

#else

#include "InstrumentedG4OpBoundaryProcess.hh"
void U4RecorderTest::UserSteppingAction(const G4Step* step){     fRecorder->UserSteppingAction<InstrumentedG4OpBoundaryProcess>(step); }

#endif




U4RecorderTest::~U4RecorderTest()
{
    G4GeometryManager::GetInstance()->OpenGeometry(); 
}

/**
G4GeometryManager::OpenGeometry is needed to avoid cleanup warning::

    WARNING - Attempt to delete the physical volume store while geometry closed !
    WARNING - Attempt to delete the logical volume store while geometry closed !
    WARNING - Attempt to delete the solid store while geometry closed !
    WARNING - Attempt to delete the region store while geometry closed !

This below constraint forces instanciating G4RunManager first in order 
to hookup physics before the main instanciation::

    You are instantiating G4UserRunAction BEFORE your G4VUserPhysicsList is
    instantiated and assigned to G4RunManager.
    Such an instantiation is prohibited by Geant4 version 8.0. To fix this problem,
    please make sure that your main() instantiates G4VUserPhysicsList AND
    set it to G4RunManager before instantiating other user action classes
    such as G4UserRunAction.

**/


