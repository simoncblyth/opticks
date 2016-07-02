#include "CFG4_BODY.hh"
#include "ActionInitialization.hh"

//#include "PrimaryGeneratorAction.hh"
//#include "G4UserSteppingAction.hh"

#include "CFG4_PUSH.hh"
#include "SteppingVerbose.hh"
#include "CFG4_POP.hh"


ActionInitialization::ActionInitialization(G4VUserPrimaryGeneratorAction* pga, G4UserSteppingAction* sa)
    : 
    G4VUserActionInitialization(), 
    m_pga(pga),
    m_sa(sa)
{}


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




