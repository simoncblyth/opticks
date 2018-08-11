#include "OpNoviceEventAction.hh"

//#include "G4Event.hh"

#ifdef WITH_OPTICKS
#include "G4Opticks.hh"
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
    G4Opticks::GetOpticks()->BeginOfEventAction(aEvent);
#endif
}

void OpNoviceEventAction::EndOfEventAction(const G4Event* aEvent)
{
#ifdef WITH_OPTICKS
    G4Opticks::GetOpticks()->EndOfEventAction(aEvent);
#endif
}

