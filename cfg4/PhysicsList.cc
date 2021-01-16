/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "CFG4_BODY.hh"


#include "globals.hh"
#include "G4OpticalPhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4Version.hh"

#if G4VERSION_NUMBER >= 1070
#else
#include "G4OpticalProcessIndex.hh"
#endif

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
