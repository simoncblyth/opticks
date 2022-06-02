#include "PLOG.hh"
#include "U4Recorder.hh"


const plog::Severity U4Recorder::LEVEL = PLOG::EnvLevel("U4Recorder", "DEBUG"); 

U4Recorder* U4Recorder::INSTANCE = nullptr ; 
U4Recorder* U4Recorder::Get(){ return INSTANCE ; }



U4Recorder::U4Recorder()
{
    INSTANCE = this ; 
}

void U4Recorder::BeginOfRunAction(const G4Run*)
{
    LOG(info); 
}
void U4Recorder::EndOfRunAction(const G4Run*)
{
    LOG(info); 
}
void U4Recorder::BeginOfEventAction(const G4Event*)
{
    LOG(info); 
}
void U4Recorder::EndOfEventAction(const G4Event*)
{
    LOG(info); 
}
void U4Recorder::PreUserTrackingAction(const G4Track*)
{
    LOG(info); 
}
void U4Recorder::PostUserTrackingAction(const G4Track*)
{
    LOG(info); 
}
void U4Recorder::UserSteppingAction(const G4Step*)
{
    LOG(info); 
}

