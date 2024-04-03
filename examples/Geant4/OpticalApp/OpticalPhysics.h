#pragma once

#include "G4VUserPhysicsList.hh"
#include "G4OpBoundaryProcess.hh"

struct OpticalPhysics : public G4VUserPhysicsList
{
    void ConstructParticle(); 
    void ConstructProcess() ; 
    void ConstructOp();
};

inline void OpticalPhysics::ConstructParticle()
{
    G4OpticalPhoton::OpticalPhotonDefinition();
}
inline void OpticalPhysics::ConstructProcess()
{
    AddTransportation();
    ConstructOp();    
}

inline void OpticalPhysics::ConstructOp()
{
    G4VProcess* boundary = new G4OpBoundaryProcess();   

    auto particleIterator=GetParticleIterator();
    particleIterator->reset();
    while( (*particleIterator)() )
    {
        G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();

        if (particleName == "opticalphoton")
        {
            pmanager->AddDiscreteProcess(boundary);
        }
    }
}

