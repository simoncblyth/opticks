#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "spho.h"
#include "SEvt.hh"
#include "PLOG.hh"

#include "U4Recorder.hh"
#include "U4Track.h"
#include "U4StepPoint.hh"
#include "U4OpBoundaryProcess.hh"
#include "G4OpBoundaryProcess.hh"
#include "U4OpBoundaryProcessStatus.h"



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
    if(U4Track::IsOptical(track)) PreUserTrackingAction_Optical(track); 
}
void U4Recorder::PostUserTrackingAction(const G4Track* track)
{ 
    if(U4Track::IsOptical(track)) PostUserTrackingAction_Optical(track); 
}
void U4Recorder::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack(); 
    if(U4Track::IsOptical(track)) UserSteppingAction_Optical(track, step); 
}

void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
{
    spho sp = U4Track::Label(track);  // just label, not sphoton 
    assert( sp.isDefined() );         // all photons are expected to be labelled, TODO: torch+input photons
    SEvt* evt = SEvt::Get(); 

    if(sp.gn == 0)
    {
        evt->beginPhoton(sp);       
    }
    else if( sp.gn > 0 )
    {
        evt->continuePhoton(sp); 
    }
}

void U4Recorder::UserSteppingAction_Optical(const G4Track* track, const G4Step* step)
{
    spho sp = U4Track::Label(track); 
    assert( sp.isDefined() );   // all photons are expected to be labelled, TODO: torch+input photons

    SEvt* evt = SEvt::Get(); 
    evt->checkPhoton(sp); 

    //unsigned status = U4OpBoundaryProcess::GetStatus() ; 
    //const char* name = U4OpBoundaryProcessStatus::Name(status) ; 
    //LOG(info) << " status " << status << " name " << name ; 

    //const G4StepPoint* pre_point = step->GetPreStepPoint() ; 
    const G4StepPoint* post_point = step->GetPostStepPoint() ; 

    sphoton& photon = evt->current_photon ;
    //photon.set_flag( flag ); 
 
    U4StepPoint::Update(photon, post_point); 
}
void U4Recorder::PostUserTrackingAction_Optical(const  G4Track* track)
{
    SEvt* evt = SEvt::Get(); 
    spho sp = U4Track::Label(track);  // just label, not sphoton 
    assert( sp.isDefined() );         // all photons are expected to be labelled, TODO: torch photons
    evt->endPhoton(sp);       
}


