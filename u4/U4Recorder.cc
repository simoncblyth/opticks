#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "spho.h"
#include "srec.h"
#include "sevent.h"

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
    SEvt* sev = SEvt::Get(); 

    if(sp.gn == 0)
    {
        sev->beginPhoton(sp);       
    }
    else if( sp.gn > 0 )
    {
        sev->continuePhoton(sp); 
    }
}


/**
U4Recorder::UserSteppingAction_Optical
---------------------------------------

*step point recording* : each step has (pre,post) and post becomes pre of next step, 
so there are two ways to record all points:

1. post-based::

   step 0: pre + post
   step 1: post 
   step 2: post
   ... 
   step n: post 

2. pre-based::

   step 0: pre
   step 1: pre 
   step 2: pre
   ... 
   step n: pre + post 

*post-based* seems preferable as truncation from various limits will complicate 
the tail of the recording. 

Q: What about reemission continuation ? 
A: The RE point should be at the same point as the AB that it scrubs, 
   so the continuing step zero should only record *post* 

HMM: need to know the step index, actually just need to know that 
are at the first step : can get that by counting bits in the flagmask, 
as only the first step will have only one bit set from the genflag 

**/

void U4Recorder::UserSteppingAction_Optical(const G4Track* track, const G4Step* step)
{
    spho sp = U4Track::Label(track); 
    assert( sp.isDefined() );   // all photons are expected to be labelled, TODO: torch+input photons

    SEvt* sev = SEvt::Get(); 
    sev->checkPhoton(sp); 

    //unsigned status = U4OpBoundaryProcess::GetStatus() ; 
    //const char* name = U4OpBoundaryProcessStatus::Name(status) ; 
    //LOG(info) << " status " << status << " name " << name ; 

    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    sphoton& photon = sev->current_photon ;

    bool single_bit = photon.flagmask_count() == 1 ; 
    // single bit genflag gets set by SEvt::beginPhoton, so the flagmask will 
    // only contain a single bit at the first step 

    if(single_bit)
    { 
        U4StepPoint::Update(photon, pre);
        sev->pointPhoton(sp); 
    }

    U4StepPoint::Update(photon, post); 

    unsigned flag = 10 ; 

    photon.set_flag( flag );
    sev->pointPhoton(sp); 
}
     



void U4Recorder::PostUserTrackingAction_Optical(const  G4Track* track)
{
    SEvt* evt = SEvt::Get(); 
    spho sp = U4Track::Label(track);  // just label, not sphoton 
    assert( sp.isDefined() );         // all photons are expected to be labelled, TODO: torch photons
    evt->endPhoton(sp);       
}


