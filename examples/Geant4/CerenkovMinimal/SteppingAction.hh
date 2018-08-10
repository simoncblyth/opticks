#pragma once

#include "G4UserSteppingAction.hh"
struct Ctx ; 

struct SteppingAction : public G4UserSteppingAction
{
    SteppingAction(Ctx* ctx_); 
    virtual void UserSteppingAction(const G4Step* step);

    Ctx* ctx ; 
};




