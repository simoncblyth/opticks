#include "G4OpticksRecorder.hh"
#include "EAction.hh"

EAction::EAction(G4OpticksRecorder* okr_)
    :
    okr(okr_)
{
}
void EAction::BeginOfEventAction(const G4Event* event)
{
    okr->BeginOfEventAction(event); 
}
void EAction::EndOfEventAction(const G4Event* event)
{
    okr->EndOfEventAction(event); 
}
