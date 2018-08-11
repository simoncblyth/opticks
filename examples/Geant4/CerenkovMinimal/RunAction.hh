#pragma once

#include "G4UserRunAction.hh"
struct Ctx ; 

struct RunAction : public G4UserRunAction
{
    RunAction(Ctx* ctx_); 
    virtual void BeginOfRunAction(const G4Run* run);
    virtual void EndOfRunAction(const G4Run* run);
    Ctx*  ctx ; 
};
