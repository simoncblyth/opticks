#pragma once

#include "G4UserTrackingAction.hh"
struct G4OpticksRecorder ; 

struct TAction : public G4UserTrackingAction
{
    TAction(G4OpticksRecorder* okr); 

    void PreUserTrackingAction(const G4Track* track);
    void PostUserTrackingAction(const G4Track* track);

    G4OpticksRecorder* okr ; 
};
