#include "G4OpticksRecorder.hh"
#include "SAction.hh"

SAction::SAction(G4OpticksRecorder* okr_)
    :
    okr(okr_)
{
}
void SAction::UserSteppingAction(const G4Step* step)
{
    okr->UserSteppingAction(step); 
}

