#include "OpNoviceEventAction.hh"

//#include "G4Event.hh"

#ifdef WITH_OPTICKS
#include "G4OpticksManager.hh"
#endif


OpNoviceEventAction::OpNoviceEventAction()
    : 
    G4UserEventAction()
{
}

OpNoviceEventAction::~OpNoviceEventAction()
{
}

void OpNoviceEventAction::BeginOfEventAction(const G4Event* aEvent)
{
#ifdef WITH_OPTICKS
  G4OpticksManager::GetOpticksManager()->BeginOfEventAction(aEvent);
#endif
}

void OpNoviceEventAction::EndOfEventAction(const G4Event* aEvent)
{
#ifdef WITH_OPTICKS
  G4OpticksManager::GetOpticksManager()->EndOfEventAction(aEvent);
#endif
}

