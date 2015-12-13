#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "SteppingAction.hh"
#include "SteppingVerbose.hh"
#include "RecorderBase.hh"

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
    SetUserAction(new PrimaryGeneratorAction(m_recorder));
    SetUserAction(new SteppingAction(m_recorder));
}

G4VSteppingVerbose* ActionInitialization::InitializeSteppingVerbose() const
{
  return new SteppingVerbose();
}  




