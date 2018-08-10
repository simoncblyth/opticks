
#include "SteppingAction.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "PLOG.hh"

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    G4String ParticleName = track->GetDynamicParticle()->GetParticleDefinition()->GetParticleName(); 
    LOG(info) << ParticleName ;  
}



