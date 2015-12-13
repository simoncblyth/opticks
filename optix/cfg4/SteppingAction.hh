#pragma once

#include "G4UserSteppingAction.hh"
#include "G4OpBoundaryProcess.hh"
#include "globals.hh"

class RecorderBase ; 

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
};

inline SteppingAction::SteppingAction(RecorderBase* recorder)
   : 
   G4UserSteppingAction(),
   m_recorder(recorder)
{ 
}

inline SteppingAction::~SteppingAction()
{ 
}



