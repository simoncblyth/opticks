
#include "G4Event.hh"

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
    std::cout << "[ EAction::EndOfEventAction " << std::endl ; 

    G4Opticks* g4ok = G4Opticks::Get(); 

    G4int eventID = event->GetEventID() ; 
    int num_hits = g4ok->propagateOpticalPhotons(eventID) ;  
    std::cout 
        << " eventID " << eventID
        << " num_hits " << num_hits 
        << std::endl
         ; 

    okr->EndOfEventAction(event);   // reset gets done in here 
    //g4ok->reset() ;  // crucially this resets the genstep collector


    std::cout << "] EAction::EndOfEventAction " << std::endl ; 
}
