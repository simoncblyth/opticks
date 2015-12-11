#include "LXeMuonPhysics.hh"

#include "globals.hh"
#include "G4ios.hh"
#include "G4PhysicalConstants.hh"
#include <iomanip>


LXeMuonPhysics::LXeMuonPhysics(const G4String& name)
                   :  G4VPhysicsConstructor(name) {
}


LXeMuonPhysics::~LXeMuonPhysics() {}


#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"

#include "G4MuonPlus.hh"
#include "G4MuonMinus.hh"
#include "G4NeutrinoMu.hh"
#include "G4AntiNeutrinoMu.hh"
#include "G4Neutron.hh"
#include "G4Proton.hh"
#include "G4PionZero.hh"
#include "G4PionPlus.hh"
#include "G4PionMinus.hh"

void LXeMuonPhysics::ConstructParticle()
{
  // Mu
  G4MuonPlus::MuonPlusDefinition();
  G4MuonMinus::MuonMinusDefinition();
  G4NeutrinoMu::NeutrinoMuDefinition();
  G4AntiNeutrinoMu::AntiNeutrinoMuDefinition();
  //These are needed for the mu- capture
    G4Neutron::Neutron();
    G4Proton::Proton();
    G4PionMinus::PionMinus();
    G4PionZero::PionZero();
    G4PionPlus::PionPlus();
}


#include "G4ProcessManager.hh"

void LXeMuonPhysics::ConstructProcess()
{
  G4MuIonisation* fMuPlusIonisation =
    new G4MuIonisation();
  G4MuMultipleScattering* fMuPlusMultipleScattering =
    new G4MuMultipleScattering();
  G4MuBremsstrahlung* fMuPlusBremsstrahlung=
    new G4MuBremsstrahlung();
  G4MuPairProduction* fMuPlusPairProduction=
    new G4MuPairProduction();

  G4MuIonisation* fMuMinusIonisation =
    new G4MuIonisation();
  G4MuMultipleScattering* fMuMinusMultipleScattering =
    new G4MuMultipleScattering();
  G4MuBremsstrahlung* fMuMinusBremsstrahlung =
    new G4MuBremsstrahlung();
  G4MuPairProduction* fMuMinusPairProduction =
    new G4MuPairProduction();

  G4MuonMinusCapture* fMuMinusCaptureAtRest =
    new G4MuonMinusCapture();

  G4ProcessManager * pManager = 0;

  // Muon Plus Physics
  pManager = G4MuonPlus::MuonPlus()->GetProcessManager();

  pManager->AddProcess(fMuPlusMultipleScattering,-1,  1, 1);
  pManager->AddProcess(fMuPlusIonisation,        -1,  2, 2);
  pManager->AddProcess(fMuPlusBremsstrahlung,    -1,  3, 3);
  pManager->AddProcess(fMuPlusPairProduction,    -1,  4, 4);

  // Muon Minus Physics
  pManager = G4MuonMinus::MuonMinus()->GetProcessManager();

  pManager->AddProcess(fMuMinusMultipleScattering,-1,  1, 1);
  pManager->AddProcess(fMuMinusIonisation,        -1,  2, 2);
  pManager->AddProcess(fMuMinusBremsstrahlung,    -1,  3, 3);
  pManager->AddProcess(fMuMinusPairProduction,    -1,  4, 4);

  pManager->AddRestProcess(fMuMinusCaptureAtRest);

}
