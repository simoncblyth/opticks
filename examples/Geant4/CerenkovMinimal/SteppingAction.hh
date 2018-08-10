#pragma once

// stepping
#include "G4UserSteppingAction.hh"
class G4Step ; 

struct SteppingAction : public G4UserSteppingAction
{
    virtual void UserSteppingAction(const G4Step* step);
};




