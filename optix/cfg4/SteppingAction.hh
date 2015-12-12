#pragma once

#include "G4UserSteppingAction.hh"
#include "G4OpBoundaryProcess.hh"
#include "globals.hh"

class RecorderBase ; 
class G4StepPoint ; 
class G4Track ; 

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(RecorderBase* recorder);
    SteppingAction();
    virtual ~SteppingAction();

    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();

    virtual void UserSteppingAction(const G4Step*);

  private:
    RecorderBase*  m_recorder ; 
    G4int fScintillationCounter;
    G4int fCerenkovCounter;
    G4int fEventNumber;
};


