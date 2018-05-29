#pragma once
#include "G4UserEventAction.hh"

class G4Event;

class OpNoviceEventAction : public G4UserEventAction
{
  public:
    OpNoviceEventAction();
    virtual ~OpNoviceEventAction();
  public:
    virtual void BeginOfEventAction(const G4Event* aEvent);
    virtual void EndOfEventAction(const G4Event* aEvent);

};

