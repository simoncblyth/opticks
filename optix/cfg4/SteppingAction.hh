#pragma once

#include "G4UserSteppingAction.hh"
#include "G4OpBoundaryProcess.hh"
#include "globals.hh"

class Recorder ; 

class SteppingAction : public G4UserSteppingAction
{
  public:
    SteppingAction(Recorder* recorder, int verbosity=0);
    virtual ~SteppingAction();

    G4OpBoundaryProcessStatus GetOpBoundaryProcessStatus();
    virtual void UserSteppingAction(const G4Step*);

  private:
    void init();

  private:
    Recorder*    m_recorder   ; 
    int          m_verbosity ; 
};

inline SteppingAction::SteppingAction(Recorder* recorder, int verbosity)
   : 
   G4UserSteppingAction(),
   m_recorder(recorder),
   m_verbosity(verbosity)
{ 
   init();
}

inline SteppingAction::~SteppingAction()
{ 
}



