

#include "globals.hh"
#include "G4OpticalPhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4OpticalProcessIndex.hh"

#include "G4SystemOfUnits.hh"

#include "PhysicsList.hh"

//  
// 
//  /usr/local/env/g4/geant4.10.02/examples/extended/optical/wls/src/WLSOpticalPhysics.cc


PhysicsList::PhysicsList() : G4VModularPhysicsList()
{
  // default cut value  (1.0mm)
  defaultCutValue = 1.0*mm;

  // follow Chroma
  //  but looking into this, far too many processes for quick tests
  //  and building the physics table takes forever
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
