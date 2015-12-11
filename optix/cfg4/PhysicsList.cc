#include "PhysicsList.hh"


#include "G4OpticalPhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4OpticalProcessIndex.hh"

#include "G4SystemOfUnits.hh"


PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
  // default cut value  (1.0mm)
  defaultCutValue = 1.0*mm;

  // follow Chroma
  RegisterPhysics( new G4EmPenelopePhysics(0) );

  G4OpticalPhysics* opticalPhysics = new G4OpticalPhysics();
  RegisterPhysics( opticalPhysics );

  //opticalPhysics->SetWLSTimeProfile("delta");
  //opticalPhysics->SetScintillationYieldFactor(1.0);
  //opticalPhysics->SetScintillationExcitationRatio(0.0);

  opticalPhysics->SetMaxNumPhotonsPerStep(100);
  opticalPhysics->SetMaxBetaChangePerStep(10.0);

  opticalPhysics->SetTrackSecondariesFirst(kCerenkov,true);
  opticalPhysics->SetTrackSecondariesFirst(kScintillation,true);

}


PhysicsList::~PhysicsList() {}


void PhysicsList::SetCuts(){
  //  " G4VUserPhysicsList::SetCutsWithDefault" method sets
  //   the default cut value for all particle types
  SetCutsWithDefault();
}
