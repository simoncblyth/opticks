/**
OpticalApp.h : Geant4 Optical Application within a single header 
===================================================================

Start from ~/o/g4cx/tests/G4CXApp.h and remove opticks dependencies for easy sharing 

Usage::

    ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh

BUT: its no so far showing the issue ? 

**/

#include <cmath>
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
    int                   UseGivenVelocity_KLUDGE ; 
    OpticalRecorder*      fRecorder ; 

    static G4Material* Vacuum(); 
    static G4MaterialPropertyVector* Make_V(double value); 
    static G4MaterialPropertiesTable* Create_MPT(const char* key, double val ); 
    static G4LogicalVolume* Box_( double halfside, G4Material* material, const char* prefix=nullptr, const double* scale=nullptr ); 

    G4VPhysicalVolume* Construct(); 

    static int getenvint(const char* ekey, int fallback); 
    static constexpr const char* _DEBUG_GENIDX = "OpticalApp__GeneratePrimaries_DEBUG_GENIDX" ;
    void GeneratePrimaries(G4Event* evt); 
    static G4PrimaryVertex* MakePrimaryVertexPhoton(int idx, int num); 

    void BeginOfRunAction(const G4Run*);
    void EndOfRunAction(const G4Run*);

    void BeginOfEventAction(const G4Event*);
    void EndOfEventAction(const G4Event*);

    void PostUserTrackingAction(const G4Track*);
    void UserSteppingAction(const G4Step*);

    static constexpr const char* _UseGivenVelocity_KLUDGE = "OpticalApp__PreUserTrackingAction_UseGivenVelocity_KLUDGE" ;
    void PreUserTrackingAction(const G4Track*);
    static bool IsOptical(const G4Track* trk);  

    static G4RunManager* InitRunManager(); 
    static int Main(); 


    OpticalApp(G4RunManager* runMgr); 
    virtual ~OpticalApp(); 
    std::string desc() const ; 
};


G4RunManager* OpticalApp::InitRunManager()  // static
{
    G4VUserPhysicsList* phy = (G4VUserPhysicsList*)new OpticalPhysics ; 
    G4RunManager* runMgr = new G4RunManager ; 
    runMgr->SetUserInitialization(phy) ; 
    return runMgr ; 
}

int OpticalApp::Main()  // static 
{
    G4RunManager* runMgr = InitRunManager(); 
    OpticalApp app(runMgr); 
    runMgr->BeamOn(1); 
    return 0 ; 
}



int OpticalApp::getenvint(const char* ekey, int fallback)
{
    char* val = getenv(ekey);
    return val ? std::atoi(val) : fallback ; 
}



OpticalApp::OpticalApp(G4RunManager* runMgr)
    :
    UseGivenVelocity_KLUDGE(getenvint(_UseGivenVelocity_KLUDGE, 0 )),
    fRecorder(new OpticalRecorder)
{
    runMgr->SetUserInitialization((G4VUserDetectorConstruction*)this);
    runMgr->SetUserAction((G4VUserPrimaryGeneratorAction*)this);
    runMgr->SetUserAction((G4UserRunAction*)this);
    runMgr->SetUserAction((G4UserEventAction*)this);
    runMgr->SetUserAction((G4UserTrackingAction*)this);
    runMgr->SetUserAction((G4UserSteppingAction*)this);
    runMgr->Initialize(); 

    fRecorder->desc = desc(); 
}

OpticalApp::~OpticalApp()
{ 
    G4GeometryManager::GetInstance()->OpenGeometry();
}

std::string OpticalApp::desc() const
{
    std::stringstream ss ; 
    ss 
       << "source:OpticalApp::desc\n"  
       << _UseGivenVelocity_KLUDGE << ":" << UseGivenVelocity_KLUDGE
       << "\n"
       ;
    std::string str = ss.str(); 
    return str ; 
}




G4Material* OpticalApp::Vacuum() // static 
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
    int DEBUG_GENIDX = getenvint(_DEBUG_GENIDX, -1 ) ;
    int ni = OpticalRecorder::MAX_PHOTON ; 

    std::cout 
        << "OpticalApp::GeneratePrimaries"
        << " ni " << ni
        << " " << _DEBUG_GENIDX << " : " << DEBUG_GENIDX
        << " (when > -1 gen only that idx) "
        << "\n"
        ;

    for(int i=0 ; i < ni ; i++)
    {
        if( DEBUG_GENIDX > -1 && i != DEBUG_GENIDX ) continue ;
        G4PrimaryVertex* vertex = MakePrimaryVertexPhoton(i, ni) ; 
        event->AddPrimaryVertex(vertex);
    }
}

inline G4PrimaryVertex* OpticalApp::MakePrimaryVertexPhoton(int idx, int num)
{
    // storch.h T_CIRCLE

    double f = double(idx)/double(num) ; // 0->~1
    double r = 50. ; 
    double azimuth_x = 0.5 ;  // restrict phi range to make semi-circle 
    double azimuth_y = 1.0 ; 
    double frac = azimuth_x*(1.-f) + azimuth_y*(f) ;  

    double phi = 2.*M_PI*frac ;
    double sinPhi = sinf(phi);
    double cosPhi = cosf(phi);

    G4ThreeVector direction(-cosPhi,0.,-sinPhi) ;   
    direction = direction.unit(); 

    G4ThreeVector position_mm(r*cosPhi,0.,r*sinPhi + r ) ;
    G4ThreeVector polarization(0.,-1.,0.) ;   // pick same as storch.h   

    polarization.rotateUz(direction);   // orient polarization 
    // HMM: unchnged

    //G4double time_ns(f)  ;
    G4double time_ns(0.)  ;   // for easier to follow animation in UseGeometryShader
    G4double wavelength_nm(420.); 

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
void OpticalApp::PostUserTrackingAction(const G4Track* trk){ fRecorder->PostUserTrackingAction(trk) ; }
void OpticalApp::UserSteppingAction(const G4Step* stp){      fRecorder->UserSteppingAction(stp) ; }        

bool OpticalApp::IsOptical(const G4Track* trk) // static 
{
    return trk->GetDefinition() == G4OpticalPhoton::OpticalPhotonDefinition() ;
}
void OpticalApp::PreUserTrackingAction(const G4Track* trk)
{  
    assert( IsOptical(trk) ); 
    std::cout << _UseGivenVelocity_KLUDGE << " : " << UseGivenVelocity_KLUDGE << "\n" ; 
    if(UseGivenVelocity_KLUDGE == 1 ) const_cast<G4Track*>(trk)->UseGivenVelocity(true);  
    fRecorder->PreUserTrackingAction(trk) ; 
}

