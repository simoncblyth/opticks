#include "LXeGeneralPhysics.hh"

#include "globals.hh"
#include "G4ios.hh"
#include <iomanip>
#include "G4Decay.hh"

LXeGeneralPhysics::LXeGeneralPhysics(const G4String& name)
                     :  G4VPhysicsConstructor(name) {}


LXeGeneralPhysics::~LXeGeneralPhysics() {
  //fDecayProcess = NULL;
}


#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"

#include "G4Geantino.hh"
#include "G4ChargedGeantino.hh"

#include "G4GenericIon.hh"

#include "G4Proton.hh"

void LXeGeneralPhysics::ConstructParticle()
{
  // pseudo-particles
  G4Geantino::GeantinoDefinition();
  G4ChargedGeantino::ChargedGeantinoDefinition();

  G4GenericIon::GenericIonDefinition();
}


void LXeGeneralPhysics::ConstructProcess()
{
  G4Decay* fDecayProcess = new G4Decay();

  // Add Decay Process
  aParticleIterator->reset();
  while( (*aParticleIterator)() ){
    G4ParticleDefinition* particle = aParticleIterator->value();
    G4ProcessManager* pmanager = particle->GetProcessManager();
    if (fDecayProcess->IsApplicable(*particle)) {
      pmanager ->AddProcess(fDecayProcess);
      // set ordering for PostStepDoIt and AtRestDoIt
      pmanager ->SetProcessOrdering(fDecayProcess, idxPostStep);
      pmanager ->SetProcessOrdering(fDecayProcess, idxAtRest);
    }
  }
}
