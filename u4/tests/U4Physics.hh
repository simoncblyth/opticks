#pragma once
/**
tests/U4Physics.hh : Not remotely reusable so consigned to tests folder together with DsG4Scintillation
=========================================================================================================

This is intended solely for use from U4RecorderTest 

**/

#include <cstdlib>
#include "G4VUserPhysicsList.hh"

class G4Cerenkov_modified ; 
class DsG4Scintillation ; 
class G4OpAbsorption ;
class G4OpRayleigh ;
class G4OpBoundaryProcess ; 


struct U4Physics : public G4VUserPhysicsList
{
    static int EInt(const char* key, const char* fallback="0"); 

    G4Cerenkov_modified*  fCerenkov ; 
    DsG4Scintillation*    fScintillation ; 
    G4OpAbsorption*       fAbsorption ;
    G4OpRayleigh*         fRayleigh ;
    G4OpBoundaryProcess*  fBoundary ;

    U4Physics(); 

    void ConstructParticle();
    void ConstructProcess();
    void ConstructEM();
    void ConstructOp();
};


U4Physics::U4Physics()
    :
    fCerenkov(nullptr),
    fScintillation(nullptr),
    fAbsorption(nullptr),
    fRayleigh(nullptr),
    fBoundary(nullptr)
{
}

#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"


inline void U4Physics::ConstructParticle()
{
    G4BosonConstructor::ConstructParticle(); 
    G4LeptonConstructor::ConstructParticle();
    G4MesonConstructor::ConstructParticle();
    G4BaryonConstructor::ConstructParticle();
    G4IonConstructor::ConstructParticle();
}

inline void U4Physics::ConstructProcess()
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

inline void U4Physics::ConstructEM()
{
    G4int em_verbosity = 0 ; 
    G4EmParameters* empar = G4EmParameters::Instance() ;
    empar->SetVerbose(em_verbosity); 
    empar->SetWorkerVerbose(em_verbosity); 

  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
    G4ParticleDefinition* particle = particleIterator->value();

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

//#include "G4Cerenkov.hh"
#include "G4Cerenkov_modified.hh"

#include "DsG4Scintillation.h"
#include "G4OpAbsorption.hh"
#include "G4OpRayleigh.hh"
#include "G4OpBoundaryProcess.hh"



inline int U4Physics::EInt(const char* key, const char* fallback)  // static 
{
    const char* val_ = getenv(key) ;
    int val =  std::atoi(val_ ? val_ : fallback) ;
    return val ; 
}

inline void U4Physics::ConstructOp()
{
    if(EInt("G4Cerenkov_DISABLE", "0") == 0 )
    {
        fCerenkov = new G4Cerenkov_modified ;
        fCerenkov->SetMaxNumPhotonsPerStep(10000);
        fCerenkov->SetMaxBetaChangePerStep(10.0);
        fCerenkov->SetTrackSecondariesFirst(true);   
        fCerenkov->SetVerboseLevel(EInt("G4Cerenkov_verboseLevel", "0"));
    }

    if(EInt("DsG4Scintillation_DISABLE", "0") == 0 )
    {
        fScintillation = new DsG4Scintillation(EInt("DsG4Scintillation_opticksMode","0")) ; 
        fScintillation->SetTrackSecondariesFirst(true);
    }


    fAbsorption = new G4OpAbsorption();
    fRayleigh = new G4OpRayleigh();
    fBoundary = new G4OpBoundaryProcess();


  auto particleIterator=GetParticleIterator();
  particleIterator->reset();
  while( (*particleIterator)() )
  {
      G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();

        if ( fCerenkov && fCerenkov->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(fCerenkov);
            pmanager->SetProcessOrdering(fCerenkov,idxPostStep);
        }

        if ( fScintillation && fScintillation->IsApplicable(*particle)) 
        {
            pmanager->AddProcess(fScintillation);
            pmanager->SetProcessOrderingToLast(fScintillation, idxAtRest);
            pmanager->SetProcessOrderingToLast(fScintillation, idxPostStep);
        }

        if (particleName == "opticalphoton") 
        {
            pmanager->AddDiscreteProcess(fAbsorption);
            pmanager->AddDiscreteProcess(fRayleigh);
            //pmanager->AddDiscreteProcess(fMieHGScatteringProcess);
            pmanager->AddDiscreteProcess(fBoundary);
        }
    }
}



