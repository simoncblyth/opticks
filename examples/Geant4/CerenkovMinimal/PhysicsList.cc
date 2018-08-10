#include "PhysicsList.hh"

#include "G4ProcessManager.hh"

// process
#include "G4VUserPhysicsList.hh"
//#include "G4Electron.hh"
//#include "G4OpticalPhoton.hh"
#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4Cerenkov.hh"
#include "G4Scintillation.hh"
#include "G4OpBoundaryProcess.hh"


#include "Cerenkov.hh"



PhysicsList::PhysicsList()
       :
       fMaxNumPhotonStep(10),
       fVerboseLevel(10),
       fCerenkovProcess(NULL),
       fScintillationProcess(NULL),
       fBoundaryProcess(NULL)
{
}


void PhysicsList::ConstructParticle()
{
    G4LeptonConstructor::ConstructParticle(); 
    G4BosonConstructor::ConstructParticle(); 
}

void PhysicsList::ConstructProcess()
{
    AddTransportation();

    fCerenkovProcess = new Cerenkov("Cerenkov");
    fCerenkovProcess->SetMaxNumPhotonsPerStep(fMaxNumPhotonStep);
    fCerenkovProcess->SetMaxBetaChangePerStep(10.0);
    fCerenkovProcess->SetTrackSecondariesFirst(true);   
    fCerenkovProcess->SetVerboseLevel(fVerboseLevel);

    fBoundaryProcess = new G4OpBoundaryProcess();

    theParticleIterator->reset();
    while( (*theParticleIterator)() )
    {
        G4ParticleDefinition* particle = theParticleIterator->value();
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
            G4cout << " AddDiscreteProcess to OpticalPhoton " << G4endl;
            //pmanager->AddDiscreteProcess(fAbsorptionProcess);
            //pmanager->AddDiscreteProcess(fRayleighScatteringProcess);
            //pmanager->AddDiscreteProcess(fMieHGScatteringProcess);
            pmanager->AddDiscreteProcess(fBoundaryProcess);
        }
    }
}



