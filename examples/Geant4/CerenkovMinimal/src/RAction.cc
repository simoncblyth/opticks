#include <vector>
#include "G4OpticksRecorder.hh"
#include "RAction.hh"

RAction::RAction(G4OpticksRecorder* okr_)
    :
    okr(okr_)
{
}

void RAction::BeginOfRunAction(const G4Run* run)
{
    okr->BeginOfRunAction(run); 
}
void RAction::EndOfRunAction(const G4Run* run)
{
    okr->EndOfRunAction(run); 
}



