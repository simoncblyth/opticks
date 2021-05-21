#include "PLOG.hh"
#include "G4OpticksRecorder.hh"


G4OpticksRecorder::G4OpticksRecorder()
{
    LOG(info); 
}

G4OpticksRecorder::~G4OpticksRecorder()
{
    LOG(info); 
}



void G4OpticksRecorder::BeginOfRunAction(const G4Run*)
{
    LOG(info); 
}
void G4OpticksRecorder::EndOfRunAction(const G4Run*)
{
    LOG(info); 
}



void G4OpticksRecorder::BeginOfEventAction(const G4Event*)
{
    LOG(info); 
}
void G4OpticksRecorder::EndOfEventAction(const G4Event*)
{
    LOG(info); 
}


void G4OpticksRecorder::PreUserTrackingAction(const G4Track*)
{
    LOG(info); 
}
void G4OpticksRecorder::PostUserTrackingAction(const G4Track*)
{
    LOG(info); 
}


void G4OpticksRecorder::UserSteppingAction(const G4Step*)
{
    LOG(info); 
}


