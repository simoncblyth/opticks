#pragma once

class RecorderBase;

#include "G4VUserActionInitialization.hh"

class ActionInitialization : public G4VUserActionInitialization
{
  public:
    ActionInitialization(RecorderBase*);
    virtual ~ActionInitialization();

    virtual void Build() const;
    virtual G4VSteppingVerbose* InitializeSteppingVerbose() const; 

  private:
    RecorderBase* m_recorder;
};


inline ActionInitialization::ActionInitialization(RecorderBase* recorder)
    : 
    G4VUserActionInitialization(), 
    m_recorder(recorder)
{}



