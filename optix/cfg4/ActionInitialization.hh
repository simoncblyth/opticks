#pragma once

class Recorder;
class TorchStepNPY ; 

#include "G4VUserActionInitialization.hh"

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(Recorder* recorder, TorchStepNPY* torch);
    virtual ~ActionInitialization();

    virtual void Build() const;
    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const; 

  private:
    Recorder* m_recorder;
    TorchStepNPY* m_torch ; 
};


inline ActionInitialization::ActionInitialization(Recorder* recorder, TorchStepNPY* torch)
    : 
    G4VUserActionInitialization(), 
    m_recorder(recorder),
    m_torch(torch)
{}



