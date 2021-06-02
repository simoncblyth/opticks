#pragma once

#include "G4UserEventAction.hh"
struct G4OpticksRecorder ; 

struct EAction : public G4UserEventAction
{
    EAction(G4OpticksRecorder* okr); 
    void BeginOfEventAction(const G4Event* event);
    void EndOfEventAction(const G4Event* event);

    G4OpticksRecorder* okr ; 
};
