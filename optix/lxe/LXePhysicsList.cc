#include "LXePhysicsList.hh"

#include "LXeGeneralPhysics.hh"
#include "LXeEMPhysics.hh"
#include "LXeMuonPhysics.hh"

#include "G4OpticalPhysics.hh"
#include "G4OpticalProcessIndex.hh"

#include "G4SystemOfUnits.hh"


LXePhysicsList::LXePhysicsList() : G4VModularPhysicsList()
{
  // default cut value  (1.0mm)
  defaultCutValue = 1.0*mm;

  // General Physics
  RegisterPhysics( new LXeGeneralPhysics("general") );

  // EM Physics
  RegisterPhysics( new LXeEMPhysics("standard EM"));

  // Muon Physics
  RegisterPhysics( new LXeMuonPhysics("muon"));

  // Optical Physics
  G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
  RegisterPhysics( opticalPhysics );

  opticalPhysics->SetWLSTimeProfile("delta");

  opticalPhysics->SetScintillationYieldFactor(1.0);
  opticalPhysics->SetScintillationExcitationRatio(0.0);

  opticalPhysics->SetMaxNumPhotonsPerStep(100);
  opticalPhysics->SetMaxBetaChangePerStep(10.0);

  opticalPhysics->SetTrackSecondariesFirst(kCerenkov,true);
  opticalPhysics->SetTrackSecondariesFirst(kScintillation,true);

}


LXePhysicsList::~LXePhysicsList() {}


void LXePhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets
  //   the default cut value for all particle types
  SetCutsWithDefault();
}
