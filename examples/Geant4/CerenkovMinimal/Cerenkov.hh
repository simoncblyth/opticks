#pragma once

#include "G4Cerenkov.hh"

struct Cerenkov : public G4Cerenkov 
{
    Cerenkov( const G4String& processName, G4ProcessType type = fElectromagnetic);

    G4bool IsApplicable(const G4ParticleDefinition& aParticleType);

    void BuildPhysicsTable(const G4ParticleDefinition& aParticleType);

    G4double PostStepGetPhysicalInteractionLength(
                                           const G4Track& aTrack,
                                           G4double ignored,
                                           G4ForceCondition* condition);

    G4VParticleChange* PostStepDoIt(const G4Track& aTrack, const G4Step& aStep);

};


