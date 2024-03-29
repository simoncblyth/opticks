#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Electron.hh"
#include "G4ParticleTable.hh"

#include "ssys.h"
#include "SEvt.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"
#include "U4.hh"
#include "U4Material.hh"

#include "DsG4Scintillation.h"


struct DsG4ScintillationTest
{
    static constexpr const char* KEY = "SCINTILLATIONYIELD" ; 
    SEvt*              evt ; 
    G4Material*        material ; 
    G4MaterialPropertiesTable* mpt ; 
    DsG4Scintillation* proc ; 
    G4VParticleChange* change  ; 

    DsG4ScintillationTest(int opticksMode);  
    void init();
    void init_SCINTILLATIONYIELD(); 
    void init_ParticleTable();
 
    void dump() const ; 
    void PostStepDoIt() ; 
    void save() const ; 
};


DsG4ScintillationTest::DsG4ScintillationTest(int opticksMode)
    :
    evt(SEvt::Create_ECPU()),
    material(U4Material::MakeScintillator()),
    mpt(material ? material->GetMaterialPropertiesTable() : nullptr), 
    proc(new DsG4Scintillation(opticksMode)),
    change(nullptr)
{
    init(); 
}
void DsG4ScintillationTest::init()
{
    init_SCINTILLATIONYIELD();
    init_ParticleTable(); 
}

void DsG4ScintillationTest::init_SCINTILLATIONYIELD()
{
    bool in_mpt = mpt->ConstPropertyExists(KEY) ; 
    double adhoc = ssys::getenvdouble(KEY, 1000.) ; 
    if(!in_mpt) mpt->AddConstProperty(KEY, adhoc ); 

    std::cout 
        << "DsG4ScintillationTest::init " 
        << " KEY " << KEY  
        << " in_mpt : " << ( in_mpt ? "YES" : "NO ") 
        ;

    if(!in_mpt) std::cout << " : ADDED ADHOC VALUE " << adhoc ; 
    std::cout << "\n" ; 
}


/**

DsG4ScintillationTest::init_ParticleTable
-----------------------------------------

Without running the below definition, G4Track::fOpticalPhoton is NULL
causing G4Track::is_opticalPhoton always false from comparing 
definition with:: 

   fOpticalPhoton = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");


BUT fpStep and fpTouchable both NULL so the track doest know 
its material so cannot do GROUPVEL lookup


**/

void DsG4ScintillationTest::init_ParticleTable()
{

    G4ParticleTable* tab = G4ParticleTable::GetParticleTable();
    assert(tab);  

    G4OpticalPhoton::OpticalPhotonDefinition();

    G4ParticleDefinition* def0 = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");
    assert(def0);  

    G4ParticleDefinition* def1 = G4OpticalPhoton::Definition() ; 
    assert(def1);  
    std::cout 
        << "DsG4ScintillationTest::init_ParticleTable"
        << " def0 " << std::hex << (uintptr_t)def0 << std::dec 
        << " def1 " << std::hex << (uintptr_t)def1 << std::dec 
        << "\n"
        ; 
    assert( def0 == def1 ); 
}

void DsG4ScintillationTest::dump() const 
{
    assert(material); 
    G4double ScintillationYield = mpt->GetConstProperty("SCINTILLATIONYIELD");
    std::cout << "ScintillationYield " << ScintillationYield << std::endl ; 


    G4PhysicsTable* slow = proc->getSlowIntegralTable();
    G4PhysicsTable* fast = proc->getFastIntegralTable();
    G4PhysicsTable* reem = proc->getReemissionIntegralTable();

    std::cout 
         << " proc " << proc << std::endl
         << " slow " << slow << std::endl
         << " fast " << fast << std::endl
         << " reem " << reem << std::endl
         ; 
    //proc->DumpPhysicsTable(); 
}


void DsG4ScintillationTest::PostStepDoIt() 
{

    G4double BetaInverse = 1.5 ; 
    G4double en = 1.*MeV ;           // HMM: what about consistency here, does it matter for scintillation ?
    G4double step_length = 1.0*mm  ; 


    G4double beta = 1./BetaInverse ; 
    G4double pre_beta = beta ; 
    G4double post_beta = beta ; 

    G4ParticleMomentum momentum(0., 0., 1.); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4Electron::Definition(),momentum);
    particle->SetPolarization(0., 0., 1. );  
    particle->SetKineticEnergy(en); 

    G4ThreeVector position(0., 0., 0.); 
    G4double time(0.); 

    G4Track* track = new G4Track(particle,time,position);

    G4StepPoint* pre = new G4StepPoint ; 
    G4StepPoint* post = new G4StepPoint ; 


    // HOW to set TouchableHandle : so the track can know material ?
    // without this cannot test setting UseGivenVelocity earlier


    G4ThreeVector pre_position(0., 0., 0.);
    G4ThreeVector post_position(0., 0., 1.);

    pre->SetPosition(pre_position); 
    post->SetPosition(post_position); 

    pre->SetVelocity(pre_beta*c_light); 
    assert( pre->GetBeta() == pre_beta );  

    post->SetVelocity(post_beta*c_light); 
    assert( post->GetBeta() == post_beta );  

    // G4Track::GetMaterial comes from current step preStepPoint 
    pre->SetMaterial(material); 
    post->SetMaterial(material); 

    G4Step* step = new G4Step ; 
    step->SetPreStepPoint(pre); 
    step->SetPostStepPoint(post); 
    step->SetStepLength(step_length); 

    G4double TotalEnergyDeposit = 1.0*MeV ;   
    step->SetTotalEnergyDeposit(TotalEnergyDeposit);  
   
    track->SetStep(step); 

    change = proc->PostStepDoIt(*track, *step) ; 
}

void DsG4ScintillationTest::save() const 
{
    NP* p = U4::CollectOpticalSecondaries(change);  
    p->save("$FOLD/p.npy"); 
    evt->saveGenstep("$FOLD"); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    int opticksMode = 3 ; 
    DsG4ScintillationTest t(opticksMode); 
    t.PostStepDoIt(); 
    t.save(); 

    return 0 ; 
}
