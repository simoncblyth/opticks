#include "G4Opticks.hh"
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

    G4Opticks* g4ok = G4Opticks::Get(); 
    g4ok->reset() ;  // crucially this resets the genstep collector

}
