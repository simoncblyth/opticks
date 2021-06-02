#pragma once

#include "G4UserRunAction.hh"
struct G4OpticksRecorder ; 

struct RAction : public G4UserRunAction
{
    RAction(G4OpticksRecorder* okr); 
    void BeginOfRunAction(const G4Run* run);
    void EndOfRunAction(const G4Run* run);

    G4OpticksRecorder*         okr ; 
};
