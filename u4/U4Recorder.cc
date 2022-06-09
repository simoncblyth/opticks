#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "spho.h"
#include "srec.h"
#include "sevent.h"

#include "SSys.hh"
#include "SEvt.hh"
#include "PLOG.hh"

#include "U4Recorder.hh"
#include "U4Track.h"
#include "U4StepPoint.hh"
#include "U4OpBoundaryProcess.hh"
#include "G4OpBoundaryProcess.hh"
#include "U4OpBoundaryProcessStatus.h"
#include "U4TrackStatus.h"



const plog::Severity U4Recorder::LEVEL = PLOG::EnvLevel("U4Recorder", "DEBUG"); 
const int U4Recorder::PIDX = SSys::getenvint("PIDX",-1) ; 
const int U4Recorder::GIDX = SSys::getenvint("GIDX",-1) ; 
bool U4Recorder::Enabled(const spho& label)
{ 
    return GIDX == -1 ? 
                        ( PIDX == -1 || label.id == PIDX ) 
                      :
                        ( GIDX == -1 || label.gs == GIDX ) 
                      ;
} 


U4Recorder* U4Recorder::INSTANCE = nullptr ; 
U4Recorder* U4Recorder::Get(){ return INSTANCE ; }

U4Recorder::U4Recorder()
{
    INSTANCE = this ; 
}

void U4Recorder::BeginOfRunAction(const G4Run*){     LOG(info); }
void U4Recorder::EndOfRunAction(const G4Run*){       LOG(info); }
void U4Recorder::BeginOfEventAction(const G4Event*){ LOG(info); }
void U4Recorder::EndOfEventAction(const G4Event*){   LOG(info); }

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



/**
U4Recorder::PreUserTrackingAction_Optical
-------------------------------------------

**Photon Labels**

Optical photon G4Track arriving here from U4 instrumented Scintillation and Cerenkov processes 
are always labelled, having the label set at the end of the generation loop by U4::GenPhotonEnd, 
see examples::

   u4/tests/DsG4Scintillation.cc
   u4/tests/G4Cerenkov_modified.cc

However primary optical photons arising from input photons or torch gensteps
are not labelled at generation as that is probably not possible without hacking 
GeneratePrimaries.

* TODO: review how torch genstep photon generation + input photon 
  worked within old workflow and bring that over to U4 
  (actually might have done this at detector framework level ?)

As a workaround for photon G4Track arriving at U4Recorder without labels, 
the spho::Fabricate method is below used to creates a label based entirely 
on a 0-based track_id with genstep index set to zero. 
This standin for a real label is only really equivalent for events 
with a single torch/input genstep. But torch gensteps are typically 
used for debugging so this restriction is ok.  

**/

void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
{
    spho label = U4Track::Label(track); 

    G4TrackStatus tstat = track->GetTrackStatus(); 
    assert( tstat == fAlive ); 

    if( label.isDefined() == false ) // happens with torch gensteps 
    {
        U4Track::SetFabricatedLabel(track); 
        label = U4Track::Label(track); 
        LOG(info) << " labelling photon " << label.desc() ; 
    }
    assert( label.isDefined() );  
    if(!Enabled(label)) return ;  

    SEvt* sev = SEvt::Get(); 

    if(label.gn == 0)
    {
        sev->beginPhoton(label);       
    }
    else if( label.gn > 0 )
    {
        sev->rjoinPhoton(label); 
    }
}

void U4Recorder::PostUserTrackingAction_Optical(const G4Track* track)
{
    spho label = U4Track::Label(track);  // just label, not sphoton 
    assert( label.isDefined() );         // all photons are expected to be labelled, TODO: input photons
    if(!Enabled(label)) return ;  

    SEvt* evt = SEvt::Get(); 
    evt->finalPhoton(label);       

    G4TrackStatus tstat = track->GetTrackStatus(); 
    if(tstat != fStopAndKill) LOG(info) << " post.tstat " << U4TrackStatus::Name(tstat) ; 
    assert( tstat == fStopAndKill ); 
}



/**
U4Recorder::UserSteppingAction_Optical
---------------------------------------

**Step Point Recording** 

Each step has (pre,post) and post becomes pre of next step, so there 
are two ways to record all points:

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


**Detecting First Step**

HMM: need to know the step index, actually just need to know that are at the first step : 
can get that by counting bits in the flagmask, as only the first step will have only 
one bit set from the genflag. The single bit genflag gets set by SEvt::beginPhoton.

**/

void U4Recorder::UserSteppingAction_Optical(const G4Track* track, const G4Step* step)
{
    spho label = U4Track::Label(track); 
    assert( label.isDefined() );   // all photons are expected to be labelled, TODO:input photons
    if(!Enabled(label)) return ;  

    G4TrackStatus tstat = track->GetTrackStatus(); 
    LOG(LEVEL) << " step.tstat " << U4TrackStatus::Name(tstat) ; 

    SEvt* sev = SEvt::Get(); 
    sev->checkPhoton(label); 

    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    sphoton& photon = sev->current_photon ;
    bool single_bit = photon.flagmask_count() == 1 ; 
    if(single_bit)
    { 
        U4StepPoint::Update(photon, pre);
        sev->pointPhoton(label);  // uses genflag set in beginPhoton
    }

    U4StepPoint::Update(photon, post); 
    //std::cout << " pre  " << U4StepPoint::Desc(pre)  << std::endl ; 
    unsigned flag = U4StepPoint::Flag(post) ; 
    if( flag == 0 ) std::cout << " ERR flag zero : post " << U4StepPoint::Desc(post) << std::endl ; 
    assert( flag > 0 ); 

    photon.set_flag( flag );
    sev->pointPhoton(label); 
}
     

