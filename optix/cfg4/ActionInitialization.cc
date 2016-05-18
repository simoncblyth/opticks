#include "ActionInitialization.hh"

//#include "PrimaryGeneratorAction.hh"
//#include "G4UserSteppingAction.hh"

#include "SteppingVerbose.hh"

ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
    SetUserAction(m_pga);
    SetUserAction(m_sa);
}

G4VSteppingVerbose* ActionInitialization::InitializeSteppingVerbose() const
{
  return new SteppingVerbose();
}  




