#pragma once

#include "G4UserSteppingAction.hh"
struct G4OpticksRecorder ; 

struct SAction : public G4UserSteppingAction
{
    SAction(G4OpticksRecorder* okr); 
    void UserSteppingAction(const G4Step* step);

    G4OpticksRecorder* okr ; 
};
