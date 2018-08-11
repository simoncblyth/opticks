#pragma once

#include "G4UserEventAction.hh"
struct Ctx ; 

struct EventAction : public G4UserEventAction
{
    EventAction(Ctx* ctx_); 
    virtual void BeginOfEventAction(const G4Event* anEvent);
    virtual void EndOfEventAction(const G4Event* anEvent);

    void addDummyHits(G4HCofThisEvent* HCE);

    Ctx*  ctx ; 
};
