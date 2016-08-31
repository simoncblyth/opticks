#include "CFG4_BODY.hh"
#include "ActionInitialization.hh"

//#include "PrimaryGeneratorAction.hh"
//#include "G4UserSteppingAction.hh"

#include "CFG4_PUSH.hh"
#include "SteppingVerbose.hh"
#include "CFG4_POP.hh"


ActionInitialization::ActionInitialization(
       G4VUserPrimaryGeneratorAction* pga, 
       G4UserSteppingAction* sa,
       G4UserRunAction* ra,
       G4UserEventAction* ea
)
    : 
    G4VUserActionInitialization(), 
    m_pga(pga),
    m_sa(sa),
    m_ra(ra),
    m_ea(ea)
{}


ActionInitialization::~ActionInitialization()
{}

void ActionInitialization::Build() const
{
    SetUserAction(m_pga);
    SetUserAction(m_sa);
    SetUserAction(m_ra);
    SetUserAction(m_ea);
}

G4VSteppingVerbose* ActionInitialization::InitializeSteppingVerbose() const
{
  return new SteppingVerbose();
}  




