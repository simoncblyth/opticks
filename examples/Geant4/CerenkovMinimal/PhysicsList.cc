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

#include "PhysicsList.hh"

#include "G4ProcessManager.hh"

// process
#include "G4VUserPhysicsList.hh"

#include "G4Version.hh"
#include "G4Scintillation.hh"
#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
#include "G4OpBoundaryProcess.hh"


template <typename T>
PhysicsList<T>::PhysicsList()
    :
    fMaxNumPhotonStep(1000),
    fVerboseLevel(1),
    fCerenkovProcess(NULL),
    fScintillationProcess(NULL),
    fBoundaryProcess(NULL)
{
}



#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"


template <typename T>
void PhysicsList<T>::ConstructParticle()
{
    G4BosonConstructor::ConstructParticle(); 
    G4LeptonConstructor::ConstructParticle(); 
    G4MesonConstructor::ConstructParticle();
    G4BaryonConstructor::ConstructParticle();
    G4IonConstructor::ConstructParticle();
}

template <typename T>
void PhysicsList<T>::ConstructProcess()
{
    AddTransportation();
    ConstructEM();
    ConstructOp();
}



// from OpNovicePhysicsList::ConstructEM

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4eMultipleScattering.hh"
#include "G4MuMultipleScattering.hh"
#include "G4hMultipleScattering.hh"

#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"

#include "G4MuIonisation.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"

#include "G4hIonisation.hh"


template <typename T>
void PhysicsList<T>::ConstructEM()
{
    G4int em_verbosity = 0 ; 
    G4EmParameters* empar = G4EmParameters::Instance() ;
    empar->SetVerbose(em_verbosity); 
    empar->SetWorkerVerbose(em_verbosity); 


#if ( G4VERSION_NUMBER >= 1042 )
  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
      G4ParticleDefinition* particle = particleIterator->value();
#else
  theParticleIterator->reset();
  while( (*theParticleIterator)() )
  {
      G4ParticleDefinition* particle = theParticleIterator->value();
#endif

    G4ProcessManager* pmanager = particle->GetProcessManager();
    G4String particleName = particle->GetParticleName();

    if (particleName == "gamma") {
    // gamma
      // Construct processes for gamma
      pmanager->AddDiscreteProcess(new G4GammaConversion());
      pmanager->AddDiscreteProcess(new G4ComptonScattering());
      pmanager->AddDiscreteProcess(new G4PhotoElectricEffect());

    } else if (particleName == "e-") {
    //electron
      // Construct processes for electron
      pmanager->AddProcess(new G4eMultipleScattering(),-1, 1, 1); 
      pmanager->AddProcess(new G4eIonisation(),       -1, 2, 2); 
      pmanager->AddProcess(new G4eBremsstrahlung(),   -1, 3, 3); 

    } else if (particleName == "e+") {
    //positron
      // Construct processes for positron
      pmanager->AddProcess(new G4eMultipleScattering(),-1, 1, 1); 
      pmanager->AddProcess(new G4eIonisation(),       -1, 2, 2); 
      pmanager->AddProcess(new G4eBremsstrahlung(),   -1, 3, 3); 
      pmanager->AddProcess(new G4eplusAnnihilation(),  0,-1, 4); 

    } else if( particleName == "mu+" ||
               particleName == "mu-"    ) { 
    //muon
     // Construct processes for muon
     pmanager->AddProcess(new G4MuMultipleScattering(),-1, 1, 1); 
     pmanager->AddProcess(new G4MuIonisation(),      -1, 2, 2); 
     pmanager->AddProcess(new G4MuBremsstrahlung(),  -1, 3, 3); 
     pmanager->AddProcess(new G4MuPairProduction(),  -1, 4, 4); 

    } else {
      if ((particle->GetPDGCharge() != 0.0) &&
          (particle->GetParticleName() != "chargedgeantino") &&
          !particle->IsShortLived()) {
       // all others charged particles except geantino
       pmanager->AddProcess(new G4hMultipleScattering(),-1,1,1);
       pmanager->AddProcess(new G4hIonisation(),       -1,2,2);
     }   
    }   
  }
}




template <typename T>
void PhysicsList<T>::ConstructOp()
{
    fCerenkovProcess = new T("Cerenkov");
    fCerenkovProcess->SetMaxNumPhotonsPerStep(fMaxNumPhotonStep);
    fCerenkovProcess->SetMaxBetaChangePerStep(10.0);
    fCerenkovProcess->SetTrackSecondariesFirst(true);   
    fCerenkovProcess->SetVerboseLevel(fVerboseLevel);

    fAbsorptionProcess = new G4OpAbsorption();
    fRayleighProcess = new G4OpRayleigh();
    fBoundaryProcess = new G4OpBoundaryProcess();


#if ( G4VERSION_NUMBER >= 1042 )
  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
      G4ParticleDefinition* particle = particleIterator->value();
#else
  theParticleIterator->reset();
  while( (*theParticleIterator)() )
  {
      G4ParticleDefinition* particle = theParticleIterator->value();
#endif
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();

        if ( fCerenkovProcess && fCerenkovProcess->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(fCerenkovProcess);
            pmanager->SetProcessOrdering(fCerenkovProcess,idxPostStep);
        }

        if ( fScintillationProcess && fScintillationProcess->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(fScintillationProcess);
            pmanager->SetProcessOrderingToLast(fScintillationProcess, idxAtRest);
            pmanager->SetProcessOrderingToLast(fScintillationProcess, idxPostStep);
        }

        if (particleName == "opticalphoton") 
        {
            G4cout << "PhysicsList<T>::ConstructOp AddDiscreteProcess to OpticalPhoton " << G4endl;
            pmanager->AddDiscreteProcess(fAbsorptionProcess);
            pmanager->AddDiscreteProcess(fRayleighProcess);
            //pmanager->AddDiscreteProcess(fMieHGScatteringProcess);
            pmanager->AddDiscreteProcess(fBoundaryProcess);
        }
    }
}



//#include "G4Cerenkov.hh"
#include "L4Cerenkov.hh"
//#include "Cerenkov.hh"

template struct PhysicsList<L4Cerenkov> ; 


