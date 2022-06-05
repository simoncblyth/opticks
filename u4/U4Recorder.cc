#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "spho.h"
#include "SEvt.hh"
#include "PLOG.hh"

#include "U4Recorder.hh"
#include "U4Track.h"


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
void U4Recorder::PreUserTrackingAction(const G4Track* track)
{
    bool op = U4Track::IsOptical(track) ; 
    spho sp = U4Track::Label(track); 

    if(sp.isDefined()) 
    {
        SEvt::AddPho(sp); 
    }
    else
    {
        if(op) LOG(fatal) << " unlabelled photon " << U4Track::Desc(track) ; 
    }
}

void U4Recorder::PostUserTrackingAction(const G4Track*)
{
    //LOG(info); 
}
void U4Recorder::UserSteppingAction(const G4Step*)
{
    //LOG(info); 
}

