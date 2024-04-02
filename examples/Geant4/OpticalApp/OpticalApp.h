/**
OpticalApp.h : Geant4 Optical Application within a single header 
===================================================================

Start from ~/o/g4cx/tests/G4CXApp.h and remove opticks dependencies for easy sharing 

**/

#include <csignal>
#include <cassert>

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
#include "G4NistManager.hh"

#include "G4Box.hh"
#include "G4PVPlacement.hh"

#include "OpticalRecorder.h"
#include "OpticalPhysics.h"

struct OpticalApp
    : 
    public G4UserRunAction,  
    public G4UserEventAction,
    public G4UserTrackingAction,
    public G4UserSteppingAction,
    public G4VUserPrimaryGeneratorAction,
    public G4VUserDetectorConstruction
{
    static std::string Desc(); 

    OpticalRecorder*      fRecorder ; 
    G4RunManager*         fRunMgr ;  
    G4VPhysicalVolume*    fPV ; 

    static G4Material* Vacuum(); 
    static G4MaterialPropertyVector* Make_V(double value); 
    static G4MaterialPropertiesTable* Create_MPT(const char* key, double val ); 
    static G4LogicalVolume* Box_( double halfside, G4Material* material, const char* prefix=nullptr, const double* scale=nullptr ); 

    G4VPhysicalVolume* Construct(); 

    void GeneratePrimaries(G4Event* evt); 
    static G4PrimaryVertex* MakePrimaryVertexPhoton(); 

    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PreUserTrackingAction(const G4Track*);
    void PostUserTrackingAction(const G4Track*);

    void UserSteppingAction(const G4Step*);

    OpticalApp(G4RunManager* runMgr); 
    static void OpenGeometry() ; 
    virtual ~OpticalApp(); 

    static G4RunManager* InitRunManager(); 
    static int           Main(); 
};


std::string OpticalApp::Desc() // static
{
    std::stringstream ss ; 
    ss << "OpticalApp::Desc" ; 
    std::string str = ss.str(); 
    return str ; 
}


OpticalApp::OpticalApp(G4RunManager* runMgr)
    :
    fRecorder(new OpticalRecorder),
    fRunMgr(runMgr),
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



G4Material* OpticalApp::Vacuum() // 
{
    G4double z = 1. ; 
    G4double a = 1.01*CLHEP::g/CLHEP::mole ; 
    G4double density = 1.00001*CLHEP::universe_mean_density ;  // curious setting to 1. gives a warning 
    G4Material* material = new G4Material("VACUUM", z, a, density );
    return material ;
}


G4MaterialPropertyVector* OpticalApp::Make_V(double value) // static
{
    int n = 2 ;
    G4double* e = new G4double[n] ;
    G4double* v = new G4double[n] ;

    e[0] = 1.55*eV ;
    e[1] = 15.5*eV ;

    v[0] = value ;
    v[1] = value ;

    G4MaterialPropertyVector* mpt = new G4MaterialPropertyVector(e, v, n);
    return mpt ;
}

G4MaterialPropertiesTable* OpticalApp::Create_MPT(const char* key, double val )  // static
{
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable ;
    G4MaterialPropertyVector* opv = Make_V(val) ;
    mpt->AddProperty(key, opv );
    return mpt ;
}

// U4VolumeMaker::Box_
G4LogicalVolume* OpticalApp::Box_( double halfside, G4Material* material, const char* prefix, const double* scale )
{
    if( prefix == nullptr ) prefix = material->GetName().c_str() ;
    double hx = scale ? scale[0]*halfside : halfside ;
    double hy = scale ? scale[1]*halfside : halfside ;
    double hz = scale ? scale[2]*halfside : halfside ;

    std::string _so = prefix ; 
    _so += "_solid" ; 
    G4Box* solid = new G4Box(_so, hx, hy, hz );

    std::string _lv = prefix ; 
    _lv += "_lv" ; 

    G4LogicalVolume* lv = new G4LogicalVolume( solid, material, _lv );
    return lv ;
}


// U4VolumeMaker::RaindropRockAirWater
G4VPhysicalVolume* OpticalApp::Construct()
{ 
    G4NistManager* nist = G4NistManager::Instance();

    G4Material* universe_material = Vacuum() ; 
    G4Material* container_material = nist->FindOrBuildMaterial("G4_Pb") ;  
    G4Material* medium_material  = nist->FindOrBuildMaterial("G4_AIR") ;
    G4Material* drop_material  = nist->FindOrBuildMaterial("G4_WATER") ;
   
    medium_material->SetMaterialPropertiesTable(Create_MPT("RINDEX",1.)) ; 
    drop_material->SetMaterialPropertiesTable(Create_MPT("RINDEX",1.333)) ; 

    double halfside = 100. ; 
    double factor = 1. ; 
    double universe_halfside = 1.2*halfside*factor ;  
    double container_halfside = 1.1*halfside*factor ;
    double medium_halfside = halfside*factor ; 
    double drop_radius = halfside/2. ;
    
    G4LogicalVolume* universe_lv  = Box_(universe_halfside, universe_material );
    G4LogicalVolume* container_lv  = Box_(container_halfside, container_material );
    G4LogicalVolume* medium_lv   = Box_(medium_halfside, medium_material );
    G4LogicalVolume* drop_lv  = Box_(drop_radius, drop_material );

    const G4VPhysicalVolume* drop_pv = new G4PVPlacement(0,G4ThreeVector(), drop_lv ,"drop_pv", medium_lv,false,0);
    const G4VPhysicalVolume* medium_pv = new G4PVPlacement(0,G4ThreeVector(),   medium_lv ,  "medium_pv",  container_lv,false,0);
    const G4VPhysicalVolume* container_pv = new G4PVPlacement(0,G4ThreeVector(),  container_lv ,  "container_pv", universe_lv,false,0);
    const G4VPhysicalVolume* universe_pv = new G4PVPlacement(0,G4ThreeVector(),  universe_lv ,  "universe_pv", nullptr,false,0);    

    assert( drop_pv );
    assert( medium_pv );
    assert( container_pv );
    assert( universe_pv );

    return const_cast<G4VPhysicalVolume*>(universe_pv) ;
}  


/**
OpticalApp::GeneratePrimaries
------------------------------------

Simplified U4VPrimaryGenerator::GeneratePrimaries_From_Photons(event, ph);

**/

void OpticalApp::GeneratePrimaries(G4Event* event)
{  
    //G4int eventID = event->GetEventID(); 
    G4PrimaryVertex* vertex = MakePrimaryVertexPhoton() ; 
    event->AddPrimaryVertex(vertex);
}

inline G4PrimaryVertex* OpticalApp::MakePrimaryVertexPhoton()
{
    G4ThreeVector position_mm(0.,0.,0.) ; 
    G4double time_ns(0.)  ;   
    G4ThreeVector direction(0.,0.,1.) ;   
    G4double wavelength_nm(420.); 
    G4ThreeVector polarization(0.,1.,0.) ;    

    G4PrimaryVertex* vertex = new G4PrimaryVertex(position_mm, time_ns);
    G4double kineticEnergy = CLHEP::h_Planck*CLHEP::c_light/(wavelength_nm*nm) ; 
    G4PrimaryParticle* particle = new G4PrimaryParticle(G4OpticalPhoton::Definition());
    particle->SetKineticEnergy( kineticEnergy );
    particle->SetMomentumDirection( direction );  
    particle->SetPolarization(polarization); 

    vertex->SetPrimary(particle);
    return vertex ; 
}

void OpticalApp::BeginOfRunAction(const G4Run* run){         fRecorder->BeginOfRunAction(run) ; } 
void OpticalApp::EndOfRunAction(const G4Run* run){           fRecorder->EndOfRunAction(run) ; }          
void OpticalApp::BeginOfEventAction(const G4Event* evt){     fRecorder->BeginOfEventAction(evt) ; } 
void OpticalApp::EndOfEventAction(const G4Event* evt){       fRecorder->EndOfEventAction(evt) ; }      
void OpticalApp::PreUserTrackingAction(const G4Track* trk){  fRecorder->PreUserTrackingAction(trk) ; }
void OpticalApp::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk) ; }
void OpticalApp::UserSteppingAction(const G4Step* stp){      fRecorder->UserSteppingAction(stp) ; }        

void OpticalApp::OpenGeometry(){  G4GeometryManager::GetInstance()->OpenGeometry(); } // static
OpticalApp::~OpticalApp(){ OpenGeometry(); }
// G4GeometryManager::OpenGeometry is needed to avoid cleanup warning

G4RunManager* OpticalApp::InitRunManager()  // static
{
    G4VUserPhysicsList* phy = (G4VUserPhysicsList*)new OpticalPhysics ; 
    G4RunManager* run = new G4RunManager ; 
    run->SetUserInitialization(phy) ; 
    return run ; 
}

int OpticalApp::Main()  // static 
{
    G4RunManager* run = InitRunManager(); 
    OpticalApp* app = new OpticalApp(run); 
    run->BeamOn(1); 
    delete app ;  // avoids "Attempt to delete the (physical volume/logical volume/solid/region) store while geometry closed" warnings 
    return 0 ; 
}

