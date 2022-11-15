#include "U4Physics.hh"
#include "G4ProcessManager.hh"

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


void U4Physics::ConstructParticle()
{
    G4BosonConstructor::ConstructParticle(); 
    G4LeptonConstructor::ConstructParticle();
    G4MesonConstructor::ConstructParticle();
    G4BaryonConstructor::ConstructParticle();
    G4IonConstructor::ConstructParticle();
}

void U4Physics::ConstructProcess()
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

void U4Physics::ConstructEM()
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

#include "Local_G4Cerenkov_modified.hh"
#include "Local_DsG4Scintillation.h"

#include "ShimG4OpAbsorption.h"
#include "ShimG4OpRayleigh.h"

#include "InstrumentedG4OpBoundaryProcess.hh"



std::string U4Physics::Desc()
{
    std::stringstream ss ; 
#ifdef DEBUG_TAG
    ss << ( ShimG4OpAbsorption::FLOAT ? "ShimG4OpAbsorption_FLOAT" : "ShimG4OpAbsorption_ORIGINAL" ) ; 
    ss << "_" ; 
    ss << ( ShimG4OpRayleigh::FLOAT ? "ShimG4OpRayleigh_FLOAT" : "ShimG4OpRayleigh_ORIGINAL" ) ; 
#endif
    std::string s = ss.str();
    return s ; 
}


int U4Physics::EInt(const char* key, const char* fallback)  // static 
{
    const char* val_ = getenv(key) ;
    int val =  std::atoi(val_ ? val_ : fallback) ;
    return val ; 
}

void U4Physics::ConstructOp()
{
    if(EInt("Local_G4Cerenkov_modified_DISABLE", "0") == 0 )
    {
        fCerenkov = new Local_G4Cerenkov_modified ;
        fCerenkov->SetMaxNumPhotonsPerStep(10000);
        fCerenkov->SetMaxBetaChangePerStep(10.0);
        fCerenkov->SetTrackSecondariesFirst(true);   
        fCerenkov->SetVerboseLevel(EInt("Local_G4Cerenkov_modified_verboseLevel", "0"));
    }

    if(EInt("Local_DsG4Scintillation_DISABLE", "0") == 0 )
    {
        fScintillation = new Local_DsG4Scintillation(EInt("Local_DsG4Scintillation_opticksMode","0")) ; 
        fScintillation->SetTrackSecondariesFirst(true);
    }


#ifdef DEBUG_TAG
    fAbsorption = new ShimG4OpAbsorption();
    fRayleigh = new ShimG4OpRayleigh();
#else
    fAbsorption = new G4OpAbsorption();
    fRayleigh = new G4OpRayleigh();
#endif

    fBoundary = new InstrumentedG4OpBoundaryProcess();


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


