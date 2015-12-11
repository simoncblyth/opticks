#pragma once

#include "G4UserSteppingAction.hh"
#include "globals.hh"

class RecorderBase ; 

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(RecorderBase* recorder);
    SteppingAction();
    virtual ~SteppingAction();

    virtual void UserSteppingAction(const G4Step*);

  private:
    RecorderBase*  m_recorder ; 
    G4int fScintillationCounter;
    G4int fCerenkovCounter;
    G4int fEventNumber;
};


