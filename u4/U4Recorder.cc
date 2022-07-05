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
#include "U4OpBoundaryProcess.h"
#include "InstrumentedG4OpBoundaryProcess.hh"
#include "U4OpBoundaryProcessStatus.h"
#include "U4TrackStatus.h"
#include "U4Random.hh"

#include "U4Process.h"


const plog::Severity U4Recorder::LEVEL = PLOG::EnvLevel("U4Recorder", "DEBUG"); 
const int U4Recorder::PIDX = SSys::getenvint("PIDX",-1) ; 
const int U4Recorder::GIDX = SSys::getenvint("GIDX",-1) ; 

std::string U4Recorder::Desc()
{
    std::stringstream ss ; 
    if( GIDX > -1 ) ss << "GIDX_" << GIDX << "_" ; 
    if( PIDX > -1 ) ss << "PIDX_" << PIDX << "_" ; 
    if( GIDX == -1 && PIDX == -1 ) ss << "ALL" ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
U4Recorder::Enabled
---------------------

This is used when PIDX and/or GIDX envvars are defined causing 
early exits from::

    U4Recorder::PreUserTrackingAction_Optical
    U4Recorder::PostUserTrackingAction_Optical
    U4Recorder::UserSteppingAction_Optical
    
Hence GIDX and PIDX provide a way to skip the expensive recording 
of other photons whilst debugging single gensteps or photons. 

**/

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
U4Recorder::U4Recorder(){ INSTANCE = this ; }

void U4Recorder::BeginOfRunAction(const G4Run*){     LOG(info); }
void U4Recorder::EndOfRunAction(const G4Run*){       LOG(info); }
void U4Recorder::BeginOfEventAction(const G4Event*){ LOG(info); }
void U4Recorder::EndOfEventAction(const G4Event*){   LOG(info); }
void U4Recorder::PreUserTrackingAction(const G4Track* track){  if(U4Track::IsOptical(track)) PreUserTrackingAction_Optical(track); }
void U4Recorder::PostUserTrackingAction(const G4Track* track){ if(U4Track::IsOptical(track)) PostUserTrackingAction_Optical(track); }
void U4Recorder::UserSteppingAction(const G4Step* step){ if(U4Track::IsOptical(step->GetTrack())) UserSteppingAction_Optical(step); }

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

* TODO: exercise torch running 
* TODO: review how input photons worked within old workflow and bring that over to U4 
  (actually might have done this at detector framework level ?)

As a workaround for photon G4Track arriving at U4Recorder without labels, 
the U4Track::SetFabricatedLabel method is below used to creates a label based entirely 
on a 0-based track_id with genstep index set to zero. This standin for a real label 
is only really equivalent for events with a single torch/input genstep. 
But torch gensteps are typically used for debugging so this restriction is ok.  

HMM: not easy to workaround this restriction as often will collect multiple gensteps 
before getting around to seeing any tracks from them so cannot devine the genstep index for a track 
by consulting gensteps collected by SEvt. YES: but this experience is from C and S gensteps, 
not torch ones so needs some experimentation to see what approach to take. 

**/

void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
{
    const_cast<G4Track*>(track)->UseGivenVelocity(true); // notes/issues/Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction.rst

    //std::cout << "U4Recorder::PreUserTrackingAction_Optical " << U4Process::Desc() << std::endl ; 
    //std::cout << U4Process::Desc() << std::endl ; 

    spho label = U4Track::Label(track); 

    G4TrackStatus tstat = track->GetTrackStatus(); 
    assert( tstat == fAlive ); 

    if( label.isDefined() == false ) // happens with torch gensteps and input photons 
    {
        U4Track::SetFabricatedLabel(track); 
        label = U4Track::Label(track); 
        LOG(LEVEL) << " labelling photon " << label.desc() ; 
    }
    assert( label.isDefined() );  
    if(!Enabled(label)) return ;  

    if(label.id % 1000 == 0 ) LOG(info) << " label.id " << label.id ; 

    U4Random::SetSequenceIndex(label.id); 

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
    spho label = U4Track::Label(track); 
    assert( label.isDefined() );  // all photons are expected to be labelled
    if(!Enabled(label)) return ;  
    U4Random::SetSequenceIndex(-1); 

    SEvt* sev = SEvt::Get(); 
    sev->finalPhoton(label);       

    G4TrackStatus tstat = track->GetTrackStatus(); 
    if(tstat != fStopAndKill) LOG(info) << " post.tstat " << U4TrackStatus::Name(tstat) ; 
    assert( tstat == fStopAndKill ); 
}

/**
U4Recorder::UserSteppingAction_Optical
---------------------------------------

**Step Point Recording** 

Each step has (pre,post) and post becomes pre of next step, so there 
are two ways to record all points. 
*post-based* seems preferable as truncation from various limits will complicate 
the tail of the recording. 

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

Q: What about reemission continuation ? 
A: The RE point should be at the same point as the AB that it scrubs,
   so the continuing step zero should only record *post* 

**Detecting First Step**

HMM: need to know the step index, actually just need to know that are at the first step.
Can get that by counting bits in the flagmask, as only the first step will have only 
one bit set from the genflag. The single bit genflag gets set by SEvt::beginPhoton
so only the first call to UserSteppingAction_Optical after PreUserTrackingAction_Optical
will fulfil *single_bit*.

**/

void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
{
    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    const G4Track* track = step->GetTrack(); 

    spho label = U4Track::Label(track); 
    assert( label.isDefined() );  
    if(!Enabled(label)) return ;  // early debug  

    //LOG(info) << " label.id " << label.id << " " << U4Process::Desc() ; 

    SEvt* sev = SEvt::Get(); 
    sev->checkPhotonLineage(label); 
    sphoton& current_photon = sev->current_ctx.p ;

    bool first_point = current_photon.flagmask_count() == 1 ;  // first_point when single bit in the flag from genflag set in beginPhoton
    if(first_point)
    { 
        U4StepPoint::Update(current_photon, pre);
        sev->pointPhoton(label);  // saves SEvt::current_photon/rec/record/prd into sevent 
    }

    unsigned flag = U4StepPoint::Flag(post) ; 
    if( flag == 0 ) LOG(error) << " ERR flag zero : post " << U4StepPoint::Desc(post) ; 
    assert( flag > 0 ); 

    unsigned boundary = 0 ;   // TODO: rustle up these 
    unsigned identity = 0 ; 

    if( flag == NAN_ABORT )
    {
        LOG(LEVEL) << " skip post saving for StepTooSmall label.id " << label.id  ;  
    }
    else
    {
        G4TrackStatus tstat = track->GetTrackStatus(); 
        Check_TrackStatus_Flag(tstat, flag); 

        U4StepPoint::Update(current_photon, post); 
        current_photon.set_flag( flag );
        current_photon.set_boundary( boundary);
        current_photon.identity = identity ; 

        sev->pointPhoton(label);         // save SEvt::current_photon/rec/seq/prd into sevent 
    }
    U4Process::ClearNumberOfInteractionLengthLeft(*track, *step); 
}

/**
//U4Track::SetStopAndKill(track); 
In CFG4 did StopAndKill but so far seems no reason to do that. Probably that was for aligning truncation.
**/

void U4Recorder::Check_TrackStatus_Flag(G4TrackStatus tstat, unsigned flag)
{
    LOG(LEVEL) << " step.tstat " << U4TrackStatus::Name(tstat) << " " << OpticksPhoton::Flag(flag)  ; 

    if( tstat == fAlive )
    {
        bool is_live_flag = OpticksPhoton::IsLiveFlag(flag);  
        if(!is_live_flag)  LOG(error) 
            << " is_live_flag " << is_live_flag 
            << " unexpected trackStatus/flag  " 
            << " trackStatus " << U4TrackStatus::Name(tstat) 
            << " flag " << OpticksPhoton::Flag(flag) 
            ;     

        assert( is_live_flag );  
    }
    else if( tstat == fStopAndKill )
    { 
        bool is_terminal_flag = OpticksPhoton::IsTerminalFlag(flag);  
        if(!is_terminal_flag)  LOG(error) 
            << " is_terminal_flag " << is_terminal_flag 
            << " unexpected trackStatus/flag  " 
            << " trackStatus " << U4TrackStatus::Name(tstat) 
            << " flag " << OpticksPhoton::Flag(flag) 
            ;     
        assert( is_terminal_flag );  
    }
    else
    {
        LOG(fatal) 
            << " unexpected trackstatus "
            << " trackStatus " << U4TrackStatus::Name(tstat) 
            << " flag " << OpticksPhoton::Flag(flag) 
            ; 
    }
}


