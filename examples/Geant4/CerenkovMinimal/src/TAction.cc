#include "G4OpticksRecorder.hh"
#include "TAction.hh"

TAction::TAction(G4OpticksRecorder* okr_)
    :
    okr(okr_)
{
}
void TAction::PreUserTrackingAction(const G4Track* track)
{
    okr->PreUserTrackingAction(track); 
}
void TAction::PostUserTrackingAction(const G4Track* track)
{
    okr->PostUserTrackingAction(track); 
}


