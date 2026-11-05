/**
G4CXThreeSpeciesTest.cc
========================

Standalone three-species (JUNO LS New Optical Model) scintillation test.

See::

    ~/o/g4cx/tests/G4CXThreeSpeciesTest.sh

**/

#include "OPTICKS_LOG.hh"
#include "G4CXApp.h"
#include "U4Material.hh"
#include "U4MaterialPropertiesTable.h"
#include "U4VolumeMaker.hh"
#include "NP.hh"

#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4MaterialPropertiesTable.hh"

#include "SLOG.hh"
#include "spath.h"

#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "Local_DsG4ScintSimple.h"

#include <map>

struct G4CXThreeSpeciesApp : public G4CXApp
{
    G4CXThreeSpeciesApp(G4RunManager* runMgr) : G4CXApp(runMgr, false) {}
    G4VPhysicalVolume* Construct() override;
    void GeneratePrimaries(G4Event* evt) override;
    static int Main();
};

/**
MakeLSMaterial
---------------

**/

static G4Material* MakeLSMaterial()
{
    G4NistManager* nist = G4NistManager::Instance();
    G4Element* H = nist->FindOrBuildElement("H");
    G4Element* C = nist->FindOrBuildElement("C");
    G4Material* LS = new G4Material("LS", 0.86*g/cm3, 2);
    LS->AddElement(C, 18);
    LS->AddElement(H, 30);

    const char* base = "$CFBaseFromGEOM/CSGFoundry/SSim/stree/material/LS";
    const char* matdir = spath::Resolve(base);

    const char* spec_keys =
        "RINDEX,GROUPVEL,RAYLEIGH,ABSLENGTH,REEMISSIONPROB,"
        "FASTCOMPONENT,SLOWCOMPONENT,"
        "LABABSLENGTH,"
        "PPOABSLENGTH,PPOREEMISSIONPROB,PPOCOMPONENT,"
        "bisMSBABSLENGTH,bisMSBREEMISSIONPROB,bisMSBCOMPONENT";
    G4MaterialPropertiesTable* mpt = U4Material::MakeMaterialPropertiesTable(base, spec_keys, ',');

    LOG_IF(fatal, mpt->GetProperty("PPOCOMPONENT") == nullptr)
        << " no PPOCOMPONENT in MPT -> check CFBaseFromGEOM envvar "
        << " matdir " << matdir
        ;

    NP* cpa = NP::Load(matdir, "MaterialConstProperty.npy");
    LOG_IF(fatal, cpa == nullptr)
        << " no MaterialConstProperty.npy in " << matdir
        << " -> geometry lacks const data (rerun geometry conversion with --new-optical-model)";
    if(cpa) {
        std::vector<std::string> ckeys;
        cpa->get_names(ckeys);
        const double* cvals = cpa->cvalues<double>();
        for (unsigned i = 0 ; i < ckeys.size() ; i++) {
            U4MaterialPropertiesTable::AddConstProperty(mpt, ckeys[i].c_str(), cvals[i]);
        }
    }

    LOG_IF(fatal, !mpt->ConstPropertyExists("LAB2PPO_PROB"))
        << " LAB2PPO_PROB missing from mpt -> MaterialConstProperty incomplete "
        << " (rerun geometry conversion with --new-optical-model)";

    LS->SetMaterialPropertiesTable(mpt);
    return LS ;
}

G4VPhysicalVolume* G4CXThreeSpeciesApp::Construct()
{
    LOG(info) << "[" ;
    G4Material* LS = MakeLSMaterial();

    // JUNO CD: LS ball rmax 17700 mm (diameter 35.4 m)
    G4double cdRadius = 17700.*mm ;
    // world box side = ball diameter + 2m  =>  halfside = radius + 1m
    G4double worldHalfside = cdRadius + 1000.*mm ;

    // inner: LS ball (U4VolumeMaker::Orb_)
    G4LogicalVolume* lsBallLV = U4VolumeMaker::Orb_(cdRadius, LS, "LSBall");

    // outer: vacuum world box
    G4Material* vac = U4Material::Vacuum("Vacuum");
    G4LogicalVolume* worldLV = U4VolumeMaker::Box_(worldHalfside, vac, "World");
    G4VPhysicalVolume* worldPV = new G4PVPlacement(nullptr, G4ThreeVector(0.,0.,0.),
        worldLV, "World_pv", nullptr, false, 0);

    // place LS ball at world centre
    U4VolumeMaker::Place(lsBallLV, worldLV);

    fPV = worldPV ;

    int imode = SEventConfig::IntegrationMode();
    if(imode == 0 || imode == 2) G4CXOpticks::SetNoSim(true);    // CPU-only: skip GPU simulator creation
    if(imode == 1 || imode == 3) G4CXOpticks::SetNoSim(false);

    G4CXOpticks::SetGeometry(fPV);
    G4CXOpticks::SaveGeometry();

    LOG(info) << "] fPV " << fPV->GetName() ;
    return worldPV ;
}

void G4CXThreeSpeciesApp::GeneratePrimaries(G4Event* evt)
{
    // e+ at 1 MeV from origin along +x -> deposits energy in LS -> ScintSimple
    G4ParticleDefinition* eplus = G4ParticleTable::GetParticleTable()->FindParticle("e+");
    LOG_IF(fatal, eplus == nullptr) << " e+ not found in particle table " ;
    assert(eplus);

    G4ParticleGun gun(1);
    gun.SetParticleDefinition(eplus);
    gun.SetParticleTime(0.0*CLHEP::ns);
    gun.SetParticlePosition(G4ThreeVector(0.*CLHEP::cm, 0.*CLHEP::cm, 0.*CLHEP::cm));
    gun.SetParticleMomentumDirection(G4ThreeVector(1., 0., 0.));
    gun.SetParticleEnergy(1.0*MeV);
    gun.GeneratePrimaryVertex(evt);
}

/**
SetBirksConstants
------------------

**/
static void SetBirksConstants()
{
    std::map<int,double> b1, b2 ;

    // e-/e+  (gamma falls back to electron, key 11)
    b1[11] = 13e-3*g/cm2/MeV ;        b2[11] = 0.0 ;
    // proton (neutron falls back to proton, key 2212)
    b1[2212] = 6.5e-3*g/cm2/MeV ;        b2[2212] = 1.5e-6*(g/cm2/MeV)*(g/cm2/MeV) ;
    // alpha
    b1[1000020040] = 3.9e-3*g/cm2/MeV ;  b2[1000020040] = 2.4e-6*(g/cm2/MeV)*(g/cm2/MeV) ;

    G4ParticleDefinition* eplus = G4ParticleTable::GetParticleTable()->FindParticle("e+");
    LOG_IF(fatal, eplus == nullptr) << " e+ not found " ;
    assert(eplus);
    G4ProcessManager* pm = eplus->GetProcessManager() ;
    LOG_IF(fatal, pm == nullptr) << " no process manager for e+ " ;

    const G4ProcessVector* plist = pm->GetProcessList() ;
    for (G4int i = 0 ; i < (G4int)plist->size() ; i++) {
        G4VProcess* proc = (*plist)[i] ;
        Local_DsG4ScintSimple* scint = dynamic_cast<Local_DsG4ScintSimple*>(proc);
        if(scint) {
            scint->SetBirksConstant1(b1);
            scint->SetBirksConstant2(b2);
            LOG(info) << "SetBirksConstants : injected on " << scint->GetProcessName() ;
        }
    }
}

int G4CXThreeSpeciesApp::Main()
{
    G4RunManager* run = InitRunManager();    // G4CXApp::InitRunManager -> new U4Physics
    G4CXThreeSpeciesApp* app = new G4CXThreeSpeciesApp(run);
    run->Initialize();
    LOG(info) << std::endl << U4Recorder::Desc();
    SetBirksConstants();                     // inject Birks constants (junosw-style, from ConstantProperty)
    app->BeamOn();
    delete app ;
    return 0 ;
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    return G4CXThreeSpeciesApp::Main();
}
