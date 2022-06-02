#include "G4RunManager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserTrackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserPhysicsList.hh"

#include "G4ParticleGun.hh"  
#include "G4Scintillation.hh"
#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4ParticleGun.hh"

#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleGun.hh"

#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"



#include "SPath.hh"
#include "U4GDML.h"
#include "OPTICKS_LOG.hh"
#include "U4Recorder.hh"

struct U4RecorderTest
    : 
    public G4UserRunAction,  
    public G4UserEventAction,
    public G4UserTrackingAction,
    public G4UserSteppingAction,
    public G4VUserPrimaryGeneratorAction,
    public G4VUserDetectorConstruction,
    public G4VUserPhysicsList
{
    static G4ParticleGun* InitGun(); 

    U4Recorder*           fRecorder ; 

    G4RunManager*         fRunMgr ;
    G4ParticleGun*        fGun ;  
    G4OpAbsorption*       fAbsorption ;
    G4OpRayleigh*         fRayleigh ;
    G4OpBoundaryProcess*  fBoundary ;

    G4VPhysicalVolume* Construct(); 
    void               ConstructParticle();
    void               ConstructProcess();

    void GeneratePrimaries(G4Event* evt); 

    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PreUserTrackingAction(const G4Track*);
    void PostUserTrackingAction(const G4Track*);

    void UserSteppingAction(const G4Step*);

    U4RecorderTest(); 
    void init(); 
};


G4ParticleGun* U4RecorderTest::InitGun()
{
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particle = particleTable->FindParticle("e+");

    G4ParticleGun* gun = new G4ParticleGun(1) ;   
    gun->SetParticleDefinition(particle);
    gun->SetParticleTime(0.0*CLHEP::ns);
    gun->SetParticlePosition(G4ThreeVector(0.0*CLHEP::cm,0.0*CLHEP::cm,0.0*CLHEP::cm));
    gun->SetParticleMomentumDirection(G4ThreeVector(1.,0.,0.));
    gun->SetParticleEnergy(0.8*MeV);   // few photons at ~0.7*MeV loads from ~ 0.8*MeV
    return gun ; 
}

U4RecorderTest::U4RecorderTest()
    :
    fRecorder(new U4Recorder),
    fRunMgr(new G4RunManager),
    fGun(InitGun()),
    fAbsorption(new G4OpAbsorption),
    fRayleigh(new G4OpRayleigh),
    fBoundary(new G4OpBoundaryProcess)
{
    init(); 
}

void U4RecorderTest::init()
{
    fRunMgr->SetUserInitialization((G4VUserDetectorConstruction*)this);
    fRunMgr->SetUserInitialization((G4VUserPhysicsList*)this); 
    fRunMgr->SetUserAction((G4VUserPrimaryGeneratorAction*)this);
    fRunMgr->SetUserAction((G4UserRunAction*)this);
    fRunMgr->SetUserAction((G4UserEventAction*)this);
    fRunMgr->SetUserAction((G4UserTrackingAction*)this);
    fRunMgr->SetUserAction((G4UserSteppingAction*)this);
    fRunMgr->Initialize(); 
}

void U4RecorderTest::ConstructParticle()
{
    G4BosonConstructor::ConstructParticle(); 
    G4LeptonConstructor::ConstructParticle();
}
void U4RecorderTest::ConstructProcess()
{
    AddTransportation();

    // TODO: bring over extracts of PhysicsList from ckm
}

G4VPhysicalVolume* U4RecorderTest::Construct()
{
    const char* gdmlpath = SPath::Resolve("$IDPath/origin_GDMLKludge.gdml", NOOP ); 
    LOG(info) << " gdmlpath " << gdmlpath ; 
    G4VPhysicalVolume* world = U4GDML::Read(gdmlpath); 
    return world ; 
}

void U4RecorderTest::GeneratePrimaries(G4Event* evt){  fGun->GeneratePrimaryVertex(evt); }


// pass along the message to the recorder
void U4RecorderTest::BeginOfRunAction(const G4Run* run){         fRecorder->BeginOfRunAction(run);   }
void U4RecorderTest::EndOfRunAction(const G4Run* run){           fRecorder->EndOfRunAction(run);     }
void U4RecorderTest::BeginOfEventAction(const G4Event* evt){     fRecorder->BeginOfEventAction(evt); }
void U4RecorderTest::EndOfEventAction(const G4Event* evt){       fRecorder->EndOfEventAction(evt);   }
void U4RecorderTest::PreUserTrackingAction(const G4Track* trk){  fRecorder->PreUserTrackingAction(trk); }
void U4RecorderTest::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk); }
void U4RecorderTest::UserSteppingAction(const G4Step* step){     fRecorder->UserSteppingAction(step); }

int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 
    U4RecorderTest t ;  
    return 0 ; 
}
