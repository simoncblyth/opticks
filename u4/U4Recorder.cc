#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "STrackInfo.h"
#include "spho.h"
#include "srec.h"

#include "NP.hh"
#include "SPath.hh"
#include "SSys.hh"
#include "SEvt.hh"
#include "SLOG.hh"
#include "SOpBoundaryProcess.hh"

#include "G4LogicalBorderSurface.hh"
#include "U4Recorder.hh"
#include "U4Engine.h"
#include "U4Track.h"
#include "U4StepPoint.hh"
#include "U4OpBoundaryProcess.h"
#include "U4OpBoundaryProcessStatus.h"
#include "U4TrackStatus.h"
#include "U4Random.hh"
#include "U4UniformRand.h"

#include "U4Surface.h"

#include "U4Process.h"

#include "SCF.h"
#include "U4Step.h"

const plog::Severity U4Recorder::LEVEL = SLOG::EnvLevel("U4Recorder", "DEBUG"); 

const int U4Recorder::STATES = SSys::getenvint("U4Recorder_STATES",-1) ; 
const int U4Recorder::RERUN  = SSys::getenvint("U4Recorder_RERUN",-1) ; 

const int U4Recorder::PIDX = SSys::getenvint("PIDX",-1) ; 
const int U4Recorder::EIDX = SSys::getenvint("EIDX",-1) ; 
const int U4Recorder::GIDX = SSys::getenvint("GIDX",-1) ; 

std::string U4Recorder::Desc()
{
    std::stringstream ss ; 
    if( GIDX > -1 ) ss << "GIDX_" << GIDX << "_" ; 
    if( EIDX > -1 ) ss << "EIDX_" << EIDX << "_" ; 
    if( GIDX == -1 && EIDX == -1 ) ss << "ALL" ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
U4Recorder::Enabled
---------------------

This is used when EIDX and/or GIDX envvars are defined causing 
early exits from::

    U4Recorder::PreUserTrackingAction_Optical
    U4Recorder::PostUserTrackingAction_Optical
    U4Recorder::UserSteppingAction_Optical
    
Hence GIDX and EIDX provide a way to skip the expensive recording 
of other photons whilst debugging single gensteps or photons. 

Formerly used PIDX rather than EIDX but that was confusing because it
is contrary to the normal use of PIDX to control debug printout for an idx. 

**/

bool U4Recorder::Enabled(const spho& label)
{ 
    return GIDX == -1 ? 
                        ( EIDX == -1 || label.id == EIDX ) 
                      :
                        ( GIDX == -1 || label.gs == GIDX ) 
                      ;
} 

U4Recorder* U4Recorder::INSTANCE = nullptr ; 
U4Recorder* U4Recorder::Get(){ return INSTANCE ; }
U4Recorder::U4Recorder()
    :
    transient_fSuspend_track(nullptr),
    rerun_rand(nullptr)
{ 
    INSTANCE = this ; 
}


void U4Recorder::BeginOfRunAction(const G4Run*){     LOG(info); }
void U4Recorder::EndOfRunAction(const G4Run*){       LOG(info); }
void U4Recorder::BeginOfEventAction(const G4Event*){ LOG(info); }
void U4Recorder::EndOfEventAction(const G4Event*){   LOG(info); }
void U4Recorder::PreUserTrackingAction(const G4Track* track){  LOG(LEVEL) ; if(U4Track::IsOptical(track)) PreUserTrackingAction_Optical(track); }
void U4Recorder::PostUserTrackingAction(const G4Track* track){ LOG(LEVEL) ; if(U4Track::IsOptical(track)) PostUserTrackingAction_Optical(track); }

template<typename T>
void U4Recorder::UserSteppingAction(const G4Step* step){ if(U4Track::IsOptical(step->GetTrack())) UserSteppingAction_Optical<T>(step); }

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
Geant4 GeneratePrimaries.

* TODO: review how input photons worked within old workflow and bring that over to U4 
  (actually might have done this at detector framework level ?)

As a workaround for photon G4Track arriving at U4Recorder without labels, 
the U4Track::SetFabricatedLabel method is below used to creates a label based entirely 
on a 0-based track_id with genstep index set to zero. This standin for a real label 
is only really equivalent for events with a single torch/inputphoton genstep. 
But torch gensteps are typically used for debugging so this restriction is ok.  

HMM: not easy to workaround this restriction as often will collect multiple gensteps 
before getting around to seeing any tracks from them so cannot devine the genstep index for a track 
by consulting gensteps collected by SEvt. YES: but this experience is from C and S gensteps, 
not torch ones so needs some experimentation to see what approach to take. 

**Reemission Rejoining**

At the tail of this method SEvt::beginPhoton OR SEvt::rejoinPhoton is called
with the spho label as argument. Which to call depends on label.gn which 
is greater than zero for reemission generations that need to be re-joined. 

HMM: can this same mechanism be used for FastSim handback to OrdinarySim ?

**/

void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
{
    bool resume_fSuspend = track == transient_fSuspend_track ; 
    G4TrackStatus tstat = track->GetTrackStatus(); 
    LOG(LEVEL) 
        << " track " << track 
        << " status:" << U4TrackStatus::Name(tstat) 
        << " resume_fSuspend " << ( resume_fSuspend ? "YES" : "NO" ) 
        ;
 
    assert( tstat == fAlive ); 
    LOG(LEVEL) << "[" ; 

    G4Track* _track = const_cast<G4Track*>(track) ; 
    _track->UseGivenVelocity(true); // notes/issues/Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction.rst

    //std::cout << "U4Recorder::PreUserTrackingAction_Optical " << U4Process::Desc() << std::endl ; 
    //std::cout << U4Process::Desc() << std::endl ; 

    spho* label = STrackInfo<spho>::GetRef(track); 

    if( label == nullptr ) // happens with torch gensteps and input photons 
    {
        int rerun_id = SEventConfig::G4StateRerun() ;
        if( rerun_id > -1 ) 
        {
            LOG(LEVEL) << " setting rerun_id " << rerun_id ; 
            U4Track::SetId(_track, rerun_id) ; 
        }

        U4Track::SetFabricatedLabel<spho>(track); 
        label = STrackInfo<spho>::GetRef(track); 
        assert(label) ; 

        LOG(LEVEL) 
            << " labelling photon :"
            << " track " << track
            << " label " << label
            << " label.desc " << label->desc() 
            ; 

        saveOrLoadStates(label->id);  // moved here as labelling happens once per torch/input photon
    }



    assert( label && label->isDefined() );  
    if(!Enabled(*label)) 
    {
       LOG(info) 
           << "NOT-enabled" 
           << " EIDX " << EIDX  
           << " GIDX " << GIDX  
           ;
       return ;  
    }

    bool modulo = label->id % 1000 == 0  ;  
    LOG_IF(info, modulo) << " modulo : label->id " << label->id ; 

    U4Random::SetSequenceIndex(label->id); 

    SEvt* sev = SEvt::Get(); 

    // Perhaps use label.gn generation for SlowSim<->FastSim transitions ?
    // BUT reemission photons can also undergo such transitions, so cannot easily reuse. 
    //
    // HMM: as this split depends only on label.gen() it could be done over in SEvt 
    if(label->gen() == 0)  
    {
        if(resume_fSuspend == false)
        {        
            sev->beginPhoton(*label);  // THIS ZEROS THE SLOT 
        }
        else  // resume_fSuspend:true happens following FastSim ModelTrigger:YES, DoIt
        {
            sev->resumePhoton(*label); 
        }
    }
    else if( label->gen() > 0 )
    {
        assert( resume_fSuspend == false ); // FastSim/SlowSim transitions of reemission photons not implemented 
        sev->rjoinPhoton(*label); 
    }
    LOG(LEVEL) << "]" ; 
}

/**
U4Recorder::saveOrLoadStates to/from NP g4state array managed by SEvt
-----------------------------------------------------------------------

Called from U4Recorder::PreUserTrackingAction_Optical

For pure-optical Geant4 photon rerunning without pre-cooked randoms
need to save the engine state into SEvt prior to using 
any Geant4 randoms for the photon. So can restore the random engine 
back to this state. 

NB for rerunning to reproduce a selected single photon this requires:

1. optical primaries have corresponding selection applied in SGenerate::GeneratePhotons
   as called from U4VPrimaryGenerator::GeneratePrimaries

2. U4Track::SetId adjusts track id to the rerun id within 
   U4Recorder::PreUserTrackingAction_Optical prior to calling this
   
**/

void U4Recorder::saveOrLoadStates( int id )  
{
    bool g4state_save = SEventConfig::IsRunningModeG4StateSave() ; 
    bool g4state_rerun = SEventConfig::IsRunningModeG4StateRerun() ; 
    bool g4state_active =  g4state_save || g4state_rerun ; 
    if( g4state_active == false ) return ; 
    
    SEvt* sev = SEvt::Get(); 

    if(g4state_save) 
    {
        NP* g4state = sev->gatherG4State(); 
        LOG_IF( fatal, g4state == nullptr ) << " cannot U4Engine::SaveState with null g4state " ; 
        assert( g4state ); 

        int max_state = g4state ? g4state->shape[0] : 0  ; 

        if( id == SEventConfig::_G4StateRerun )
        {
            LOG(LEVEL) 
                << "U4Engine::SaveState for id (SEventConfig::_G4StateRerun) " << id 
                << " from the max_state " << max_state 
                << std::endl 
                << U4Engine::DescStateArray()
                ; 
        }
        U4Engine::SaveState( g4state, id );      
    }
    else if(g4state_rerun)
    {
        const NP* g4state = sev->getG4State(); 
        LOG_IF( fatal, g4state == nullptr ) << " cannot U4Engine::RestoreState with null g4state " ; 
        assert( g4state ); 

        U4Engine::RestoreState( g4state, id );   


        if( id == SEventConfig::_G4StateRerun )
        {
            rerun_rand = U4UniformRand::Get(1000);
            U4UniformRand::UU = rerun_rand ; 
            SEvt::UU = rerun_rand ;  // better hitching it somewhere thats always accessible 

            LOG(LEVEL) 
                << "U4Engine::RestoreState for id (SEventConfig::_G4StateRerun)  " << id 
                << std::endl 
                << U4Engine::DescStateArray()
                << std::endl
                << " rerun_rand (about to be consumed, did RestoreState after collecting them)  "
                << std::endl
                << rerun_rand->repr<double>()  
                << std::endl
                ; 

            U4Engine::RestoreState( g4state, id );   

        }
    }
}

void U4Recorder::saveRerunRand(const char* dir) const 
{
    if( rerun_rand == nullptr ) return ; 

    int id = SEventConfig::_G4StateRerun ; 
    std::string name = U::FormName("U4Recorder_G4StateRerun_", id, ".npy" ); 
    rerun_rand->save( dir, name.c_str()); 
}

/**
U4Recorder::PostUserTrackingAction_Optical

**/

void U4Recorder::PostUserTrackingAction_Optical(const G4Track* track)
{
    LOG(LEVEL) << "[" ; 

    G4TrackStatus tstat = track->GetTrackStatus(); 
    LOG(LEVEL) << U4TrackStatus::Name(tstat) ; 

    bool is_fStopAndKill = tstat == fStopAndKill ; 
    bool is_fSuspend     = tstat == fSuspend ; 
    bool is_fStopAndKill_or_fSuspend = is_fStopAndKill || is_fSuspend  ; 
    LOG_IF(info, !is_fStopAndKill_or_fSuspend ) << " not is_fStopAndKill_or_fSuspend  post.tstat " << U4TrackStatus::Name(tstat) ; 
    assert( is_fStopAndKill_or_fSuspend ); 


    spho* label = STrackInfo<spho>::GetRef(track); 
    assert( label && label->isDefined() );  // all photons are expected to be labelled
    if(!Enabled(*label)) return ;  

    SEvt* sev = SEvt::Get(); 

    if(is_fStopAndKill)
    {
        U4Random::SetSequenceIndex(-1); 
        sev->finalPhoton(*label);  
        transient_fSuspend_track = nullptr ;

#ifndef PRODUCTION
        sseq& seq = sev->current_ctx.seq ; 
        LOG(info) 
            << " label.id " << std::setw(5) << label->id
            << " seq.desc_seqhis " << seq.desc_seqhis()
            ;  
#endif


    }
    else if(is_fSuspend)
    {
        transient_fSuspend_track = track ; 
    }


    LOG(LEVEL) << "]" ; 
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

HMM: but if subsequent step points failed to set a non-zero flag could get that repeated 

**/

template <typename T>
void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
{
    const G4Track* track = step->GetTrack(); 
    G4VPhysicalVolume* pv = track->GetVolume() ; 
    LOG(LEVEL) << "[ pv " << ( pv ? pv->GetName() : "-" ) ; 

    spho* label = STrackInfo<spho>::GetRef(track); 
    assert( label->isDefined() );  
    if(!Enabled(*label)) return ;  

    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    SEvt* sev = SEvt::Get(); 
    sev->checkPhotonLineage(*label); 
    sphoton& current_photon = sev->current_ctx.p ;
    quad4& current_aux = sev->current_ctx.aux ; 

    SOpBoundaryProcess* bop = SOpBoundaryProcess::Get(); 
    current_aux.q0.f.x = bop->getU0() ; 
    current_aux.q0.i.w = bop->getU0_idx() ; 


    // first_point when single bit in the flag from genflag set in beginPhoton
    bool first_point = current_photon.flagmask_count() == 1 ;  
    if(first_point)
    { 
        LOG(LEVEL) << " first_point, track " << track  ; 
        U4StepPoint::Update(current_photon, pre);
        sev->pointPhoton(*label);  // saves SEvt::current_photon/rec/record/prd into sevent 
    }

    unsigned flag = U4StepPoint::Flag<T>(post) ; 

    // DEFER_FSTRACKINFO : special flag signalling that 
    // the FastSim DoIt status needs to be accessed via the 
    // trackinfo label 

    if(flag == DEFER_FSTRACKINFO)
    {
        char fstrackinfo_stat = label->uc4.w ; 
        label->uc4.w = '_' ;   // scrub after access 

        switch(fstrackinfo_stat)
        {
           case 'T': flag = BOUNDARY_TRANSMIT ; break ; 
           case 'R': flag = BOUNDARY_REFLECT  ; break ; 
           case 'A': flag = SURFACE_ABSORB    ; break ; 
           case 'D': flag = SURFACE_DETECT    ; break ; 
           case '_': flag = 0                 ; break ; 
           default:  flag = 0                 ; break ; 
        }
        LOG_IF(error, flag == 0)
            << " DEFER_FSTRACKINFO " 
            << " FAILED TO GET THE FastSim status from trackinfo " 
            << " fstrackinfo_stat " << fstrackinfo_stat  
            ;

        LOG(LEVEL) 
            << " DEFER_FSTRACKINFO " 
            << " fstrackinfo_stat " << fstrackinfo_stat 
            << " flag " << OpticksPhoton::Flag(flag) 
            ; 
    }

    LOG_IF(error, flag == 0) << " ERR flag zero : post " << U4StepPoint::Desc<T>(post) ; 
    assert( flag > 0 ); 

    LOG(LEVEL) << U4StepPoint::DescPositionTime(post) ;  


    if( flag == NAN_ABORT )
    {
        LOG(LEVEL) << " skip post saving for StepTooSmall label.id " << label->id  ;  
    }
    else
    {
        G4TrackStatus tstat = track->GetTrackStatus(); 

        Check_TrackStatus_Flag(tstat, flag, "UserSteppingAction_Optical" ); 

        U4StepPoint::Update(current_photon, post); 

        current_photon.set_flag( flag );

        if(U4Step::CF) U4Step::MockOpticksBoundaryIdentity(current_photon, step, label->id ); 

        sev->pointPhoton(*label);         // save SEvt::current_photon/rec/seq/prd into sevent 
    }


    U4Process::ClearNumberOfInteractionLengthLeft(*track, *step); 
    LOG(LEVEL) << "]" ; 
}





/**
//U4Track::SetStopAndKill(track); 
In CFG4 did StopAndKill but so far seems no reason to do that. Probably that was for aligning truncation.
**/

void U4Recorder::Check_TrackStatus_Flag(G4TrackStatus tstat, unsigned flag, const char* from )
{
    LOG(LEVEL) << " step.tstat " << U4TrackStatus::Name(tstat) << " " << OpticksPhoton::Flag(flag) << " from " << from  ; 

    if( tstat == fAlive )
    {
        bool is_live_flag = OpticksPhoton::IsLiveFlag(flag);    // with FastSim seeing fSuspend
        LOG_IF(error, !is_live_flag ) 
            << " is_live_flag " << is_live_flag 
            << " unexpected trackStatus/flag  " 
            << " trackStatus " << U4TrackStatus::Name(tstat) 
            << " flag " << OpticksPhoton::Flag(flag) 
            ;     

        assert( is_live_flag );  
    }
    else if( tstat == fSuspend )
    {
        LOG(LEVEL) << " fSuspend : this happens when hand over to FastSim, ie when ModelTrigger:YES " ;  
    }
    else if( tstat == fStopAndKill )
    { 
        bool is_terminal_flag = OpticksPhoton::IsTerminalFlag(flag);  
        LOG_IF(error, !is_terminal_flag ) 
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


#include "InstrumentedG4OpBoundaryProcess.hh"

template void U4Recorder::UserSteppingAction<InstrumentedG4OpBoundaryProcess>(const G4Step*) ; 
template void U4Recorder::UserSteppingAction_Optical<InstrumentedG4OpBoundaryProcess>(const G4Step*) ; 


