#pragma once

#include "G4UserSteppingAction.hh"
#include "G4OpBoundaryProcess.hh"
#include "globals.hh"

class Recorder ; 

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(Recorder* recorder);
    virtual ~SteppingAction();

    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
    virtual void UserSteppingAction(const G4Step*);

  private:
    Recorder*  m_recorder ; 
};

inline SteppingAction::SteppingAction(Recorder* recorder)
   : 
   G4UserSteppingAction(),
   m_recorder(recorder)
{ 
}

inline SteppingAction::~SteppingAction()
{ 
}



