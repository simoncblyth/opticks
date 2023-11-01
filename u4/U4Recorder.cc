#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "STrackInfo.h"
#include "spho.h"
#include "srec.h"
#include "ssys.h"
#include "stimer.h"
#include "smeta.h"

#include "NP.hh"
#include "SPath.hh"
#include "ssys.h"
#include "ssolid.h"
#include "SEvt.hh"
#include "SLOG.hh"
#include "SOpBoundaryProcess.hh"
#include "CustomStatus.h"

#include "G4LogicalBorderSurface.hh"
#include "G4Event.hh"

#include "U4Recorder.hh"
#include "U4Engine.h"
#include "U4Track.h"
#include "U4StepPoint.hh"
#include "U4OpBoundaryProcess.h"
#include "U4OpBoundaryProcessStatus.h"
#include "U4TrackStatus.h"
#include "U4Random.hh"
#include "U4UniformRand.h"
#include "U4Fake.h"

#include "U4Surface.h"
#include "U4Process.h"
#include "U4Touchable.h"
#include "U4Simtrace.h"

#include "SCF.h"
#include "U4Step.h"


#include "G4OpBoundaryProcess.hh"
#ifdef WITH_CUSTOM4
#include "C4OpBoundaryProcess.hh" 
#include "C4CustomART.h" 
#include "C4CustomART_Debug.h" 
#include "C4TrackInfo.h"
#include "C4Pho.h"
#include "C4Version.h"
#elif PMTSIM_STANDALONE
#include "CustomART.h" 
#include "CustomART_Debug.h" 
#endif


const plog::Severity U4Recorder::LEVEL = SLOG::EnvLevel("U4Recorder", "DEBUG"); 
UName                U4Recorder::SPECS = {} ; 

const int U4Recorder::STATES = ssys::getenvint("U4Recorder_STATES",-1) ; 
const int U4Recorder::RERUN  = ssys::getenvint("U4Recorder_RERUN",-1) ; 

const bool U4Recorder::PIDX_ENABLED = ssys::getenvbool("U4Recorder__PIDX_ENABLED") ; 
const bool U4Recorder::EndOfRunAction_Simtrace = ssys::getenvbool("U4Recorder__EndOfRunAction_Simtrace") ; 
const char* U4Recorder::REPLICA_NAME_SELECT = ssys::getenvvar("U4Recorder__REPLICA_NAME_SELECT", "PMT") ;  


const int U4Recorder::PIDX = ssys::getenvint("PIDX",-1) ; 
const int U4Recorder::EIDX = ssys::getenvint("EIDX",-1) ; 
const int U4Recorder::GIDX = ssys::getenvint("GIDX",-1) ; 

std::string U4Recorder::Desc() // static
{
    std::stringstream ss ; 
    ss << "U4Recorder::Desc" << std::endl 
       << " U4Recorder_STATES                   : " << STATES << std::endl 
       << " U4Recorder_RERUN                    : " << RERUN << std::endl 
       << " U4Recorder__PIDX_ENABLED            : " << ( PIDX_ENABLED            ? "YES" : "NO " ) << std::endl 
       << " U4Recorder__EndOfRunAction_Simtrace : " << ( EndOfRunAction_Simtrace ? "YES" : "NO " ) << std::endl  
       << " U4Recorder__REPLICA_NAME_SELECT     : " << ( REPLICA_NAME_SELECT     ? REPLICA_NAME_SELECT : "-" ) << std::endl 
       << " PIDX                                : " << PIDX << std::endl 
       << " EIDX                                : " << EIDX << std::endl 
       << " GIDX                                : " << GIDX << std::endl 
       ;

    bool uoc = UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft ; 
    ss << UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft_ << ":" << int(uoc) << std::endl ; 
    ss << Switches() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

/**
U4Recorder::Switches
---------------------

HMM: preping metadata kvs would be more useful 
**/

std::string U4Recorder::Switches()  // static 
{
    std::stringstream ss ; 
    ss << "U4Recorder::Switches" << std::endl ; 
#ifdef WITH_CUSTOM4
    ss << "WITH_CUSTOM4" << std::endl ; 
#else
    ss << "NOT:WITH_CUSTOM4" << std::endl ; 
#endif
#ifdef WITH_PMTSIM
    ss << "WITH_PMTSIM" << std::endl ; 
#else
    ss << "NOT:WITH_PMTSIM" << std::endl ; 
#endif
#ifdef PMTSIM_STANDALONE
    ss << "PMTSIM_STANDALONE" << std::endl ; 
#else
    ss << "NOT:PMTSIM_STANDALONE" << std::endl ; 
#endif

#ifdef PRODUCTION
    ss << "PRODUCTION" << std::endl ; 
#else
    ss << "NOT:PRODUCTION" << std::endl ; 
#endif

    std::string str = ss.str(); 
    return str ; 
}




std::string U4Recorder::EnabledLabel() // static   (formerly Desc)
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

This is typically used only during early stage debugging. 

**/

bool U4Recorder::Enabled(const spho& label)
{ 
    return GIDX == -1 ? 
                        ( EIDX == -1 || label.id == EIDX ) 
                      :
                        ( GIDX == -1 || label.gs == GIDX ) 
                      ;
} 


std::string U4Recorder::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Recorder::desc" ; 
    std::string s = ss.str(); 
    return s ; 
}



U4Recorder* U4Recorder::INSTANCE = nullptr ; 
U4Recorder* U4Recorder::Get(){ return INSTANCE ; }


/**
U4Recorder::U4Recorder
-----------------------

CAUTION: LSExpDetectorConstruction_Opticks::Setup for opticksMode:2 
will instanciate SEvt if this is not used 

**/

U4Recorder::U4Recorder()
    :
    eventID(-1),
    transient_fSuspend_track(nullptr),
    rerun_rand(nullptr),
    sev(SEvt::HighLevelCreate(SEvt::ECPU))   
{
    init();  
}

void U4Recorder::init()
{
    INSTANCE = this ; 
    init_SEvt(); 
}

void U4Recorder::init_SEvt()
{
    smeta::Collect(sev->meta, "U4Recorder::init_SEvt" ); 

#ifdef WITH_CUSTOM4
    NP::SetMeta<std::string>(sev->meta, "C4Version", C4Version::Version() ); 
#else
    NP::SetMeta<std::string>(sev->meta, "C4Version", "NOT-WITH_CUSTOM4" ); 
#endif

    LOG(LEVEL) << " sev " << std::hex << sev << std::dec ; 
    LOG(LEVEL) << "sev->meta[" << sev->meta << "]" ;
}



void U4Recorder::BeginOfRunAction(const G4Run*)
{  
    SEvt::BeginOfRun(); // just sets static stamp 
    LOG(info); 
}

/**
U4Recorder::EndOfRunAction
---------------------------

HUH: some problem with LOG from here ? 

**/
void U4Recorder::EndOfRunAction(const G4Run*)
{ 
    SEvt::EndOfRun();  // just sets static stamp

    SEvt::SetRunMeta<int>("FAKES_SKIP", int(FAKES_SKIP) ); 
    SEvt::SaveRunMeta(); 

    LOG(LEVEL)
        << "[ U4Recorder__EndOfRunAction_Simtrace : " << ( EndOfRunAction_Simtrace ? "YES" : "NO " )  
        ;
 
    if(EndOfRunAction_Simtrace) 
    {
        U4Simtrace::EndOfRunAction() ; 
    }

    LOG(LEVEL)
        << "] U4Recorder__EndOfRunAction_Simtrace : " << ( EndOfRunAction_Simtrace ? "YES" : "NO " )  
        ;


}





void U4Recorder::BeginOfEventAction(const G4Event* event)
{ 
    eventID = event->GetEventID() ; 
    LOG(info) << " eventID " << eventID ; 

    sev->beginOfEvent(eventID);  
}

void U4Recorder::EndOfEventAction(const G4Event* event)
{  
    G4int eventID_ = event->GetEventID() ; 
    assert( eventID == eventID_ ); 

    sev->add_array("U4R.npy", MakeMetaArray() ); 
    sev->addEventConfigArray(); 

    sev->endOfEvent(eventID_);  // does save and clear

    const char* savedir = sev->getSaveDir() ; 
    LOG(info) << " savedir " << ( savedir ? savedir : "-" );
    SaveMeta(savedir);  

}
void U4Recorder::PreUserTrackingAction(const G4Track* track){  LOG(LEVEL) ; if(U4Track::IsOptical(track)) PreUserTrackingAction_Optical(track); }
void U4Recorder::PostUserTrackingAction(const G4Track* track){ LOG(LEVEL) ; if(U4Track::IsOptical(track)) PostUserTrackingAction_Optical(track); }



#include "U4OpBoundaryProcess.h"

void U4Recorder::UserSteppingAction(const G4Step* step)
{ 
    if(!U4Track::IsOptical(step->GetTrack())) return ; 

#if defined(WITH_CUSTOM4)
     UserSteppingAction_Optical<C4OpBoundaryProcess>(step); 
#elif defined(WITH_PMTSIM)
     UserSteppingAction_Optical<CustomG4OpBoundaryProcess>(step); 
#else
     UserSteppingAction_Optical<InstrumentedG4OpBoundaryProcess>(step);
#endif
}


/**
U4Recorder::PreUserTrackingAction_Optical
-------------------------------------------

1. access photon label from the G4Track
   or fabricate the label if there is none. 
   Scintillation and Cerenkov photons should always already 
   have labels associated, but torch gensteps or input photons 
   do not thus labels are created and associated here. 
 
2. photon label is passed along to the appropriate SEvt methods, 
   such as beginPhoton, resumePhoton, rjoinPhoton, rjoin_resumePhoton


**Reemission Rejoining**

At the tail of this method SEvt::beginPhoton OR SEvt::rejoinPhoton is called
with the spho label as argument. Which to call depends on label.gn which 
is greater than zero for reemission generations that need to be re-joined. 

HMM: can this same mechanism be used for FastSim handback to OrdinarySim ?

**/

void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
{
    LOG(LEVEL) << "[" ; 

    G4Track* _track = const_cast<G4Track*>(track) ; 
    _track->UseGivenVelocity(true); // notes/issues/Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction.rst

    spho ulabel = {} ; 
    PreUserTrackingAction_Optical_GetLabel(ulabel, track); 

    bool skip = !Enabled(ulabel) ; 
    LOG_IF( info, skip ) << " Enabled-SKIP  EIDX/GIDX " << EIDX << "/" << GIDX ;  
    if(skip) return ; 

    bool modulo = ulabel.id % 1000 == 0  ;  
    LOG_IF(info, modulo) << " modulo 1000 : ulabel.id " << ulabel.id ; 

    U4Random::SetSequenceIndex(ulabel.id); 

    bool resume_fSuspend = track == transient_fSuspend_track ; 
    G4TrackStatus tstat = track->GetTrackStatus(); 
    LOG(LEVEL) 
        << " track " << track 
        << " status:" << U4TrackStatus::Name(tstat) 
        << " resume_fSuspend " << ( resume_fSuspend ? "YES" : "NO" ) 
        ;
    assert( tstat == fAlive ); 

    SEvt* sev = SEvt::Get_ECPU(); 
    LOG_IF(fatal, sev == nullptr) << " SEvt::Get(1) returned nullptr " ; 
    assert(sev); 

    if(ulabel.gen() == 0)  
    {
        if(resume_fSuspend == false)
        {        
            sev->beginPhoton(ulabel);  // THIS ZEROS THE SLOT 
        }
        else  // resume_fSuspend:true happens following FastSim ModelTrigger:YES, DoIt
        {
            sev->resumePhoton(ulabel); 
        }
    }
    else if( ulabel.gen() > 0 )   // HMM: thats going to stick for reemission photons 
    {
        if(resume_fSuspend == false)
        {
            sev->rjoinPhoton(ulabel); 
        }
        else   // resume_fSuspend:true happens following FastSim ModelTrigger:YES, DoIt
        {
            sev->rjoin_resumePhoton(ulabel); 
        }
    }
    LOG(LEVEL) << "]" ; 
    //std::cout << "] U4Recorder::PreUserTrackingAction_Optical " << std::endl ;
}

/**
U4Recorder::PreUserTrackingAction_Optical_GetLabel
---------------------------------------------------

Optical photon G4Track arriving here from U4 instrumented Scintillation and Cerenkov processes 
are always labelled, having the label set at the end of the generation loop by U4::GenPhotonEnd, 
see examples::

   u4/tests/DsG4Scintillation.cc
   u4/tests/G4Cerenkov_modified.cc

However primary optical photons arising from input photons or torch gensteps
are not labelled at generation as that is probably not possible without hacking 
Geant4 GeneratePrimaries.

As a workaround for photon G4Track arriving at U4Recorder without labels, 
the U4Track::SetFabricatedLabel method is below used to creates a label based entirely 
on a 0-based track_id with genstep index set to zero. This standin for a real label 
is only really equivalent for events with a single torch/inputphoton genstep. 
But torch gensteps are typically used for debugging so this restriction is ok.  

HMM: not easy to workaround this restriction as often will collect multiple gensteps 
before getting around to seeing any tracks from them so cannot devine the genstep index for a track 
by consulting gensteps collected by SEvt. YES: but this experience is from C and S gensteps, 
not torch ones so needs some experimentation to see what approach to take. 

**/

void U4Recorder::PreUserTrackingAction_Optical_GetLabel( spho& ulabel, const G4Track* track )
{
#ifdef WITH_CUSTOM4
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
#else
    spho* label = STrackInfo<spho>::GetRef(track); 
#endif

    if( label == nullptr ) // happens with torch gensteps and input photons 
    {
        PreUserTrackingAction_Optical_FabricateLabel(track) ; 
#ifdef WITH_CUSTOM4
        label = C4TrackInfo<C4Pho>::GetRef(track); 
#else
        label = STrackInfo<spho>::GetRef(track); 
#endif
    }
    assert( label && label->isDefined() );  

#ifdef WITH_CUSTOM4
    assert( C4Pho::N == spho::N ); 
#endif
    std::array<int,spho::N> a_label ; 
    label->serialize(a_label) ; 

    ulabel.load(a_label); 

    // serialize/load provides firebreak between C4Pho and spho
    // so the SEvt doesnt need to depend on CUSTOM4
}

void U4Recorder::PreUserTrackingAction_Optical_FabricateLabel( const G4Track* track )
{
    int rerun_id = SEventConfig::G4StateRerun() ;
    if( rerun_id > -1 ) 
    {
        LOG(LEVEL) << " setting rerun_id " << rerun_id ; 
        G4Track* _track = const_cast<G4Track*>(track) ; 
        U4Track::SetId(_track, rerun_id) ; 
    }

#ifdef WITH_CUSTOM4
    U4Track::SetFabricatedLabel<C4Pho>(track); 
#else
    U4Track::SetFabricatedLabel<spho>(track); 
#endif

#ifdef WITH_CUSTOM4
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
#else
    spho* label = STrackInfo<spho>::GetRef(track); 
#endif
    assert(label) ; 

    LOG(LEVEL) 
        << " labelling photon :"
        << " track " << track
        << " label " << label
        << " label.desc " << label->desc() 
        ; 

    saveOrLoadStates(label->id);  // moved here as labelling happens once per torch/input photon
}

/**
U4Recorder::GetLabel
----------------------

Unlike the above PreUserTrackingAction_Optical_GetLabel 
this does not handle the case of the track not being labelled. 

**/

void U4Recorder::GetLabel( spho& ulabel, const G4Track* track )
{
#ifdef WITH_CUSTOM4
    C4Pho* label = C4TrackInfo<C4Pho>::GetRef(track); 
#else
    spho* label = STrackInfo<spho>::GetRef(track); 
#endif
    assert( label && label->isDefined() && "all photons are expected to be labelled" ); 

#ifdef WITH_CUSTOM4
    assert( C4Pho::N == spho::N ); 
#endif
    std::array<int,spho::N> a_label ; 
    label->serialize(a_label) ; 

    ulabel.load(a_label); 
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
    bool first_event = eventID == 0 ; 
    LOG_IF(LEVEL, !first_event ) << " skip as not first_event eventID " << eventID ; 
    if(!first_event) return ; 

    bool g4state_save = SEventConfig::IsRunningModeG4StateSave() ; 
    bool g4state_rerun = SEventConfig::IsRunningModeG4StateRerun() ; 
    bool g4state_active =  g4state_save || g4state_rerun ; 
    if( g4state_active == false ) return ; 
    
    SEvt* sev = SEvt::Get_ECPU(); 

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

/**
U4Recorder::saveRerunRand
---------------------------

TODO: adopt SEvt::add_array 

**/


void U4Recorder::saveRerunRand(const char* dir) const 
{
    if( rerun_rand == nullptr ) return ; 

    int id = SEventConfig::_G4StateRerun ; 
    std::string name = U::FormName("U4Recorder_G4StateRerun_", id, ".npy" ); 
    rerun_rand->save( dir, name.c_str()); 
}

/**
U4Recorder::SaveMeta
----------------------

This is called from U4Recorder::EndOfEventAction after saving the SEvt 
in order to have savedir. 

SPECS.names 
    added and enumerations used in UserSteppingAction_Optical
    enumeration spec values at each point a.f.aux[:,:,2,3].view(np.int32) 

With standalone testing this is called from U4App::EndOfEventAction/U4App::SaveMeta

HMM this is saving extra metadata onto the SEvt savedir.
Could instead add SEvt API for addition of metadata, 
avoiding need to passaround savedir.
This would make it easier to add SEvt metadata 
from within the monolith as just need access to SEvt
and dont need to catch it after the save. 

**/

void U4Recorder::SaveMeta(const char* savedir)  // static
{
    if(savedir == nullptr) return ; 

    assert(INSTANCE); 
    INSTANCE->saveRerunRand(savedir); 
}   


NP* U4Recorder::MakeMetaArray() // static
{
    std::string descfakes = U4Recorder::DescFakes() ;
    LOG(info) << descfakes ; 

    NP* u4r = NP::Make<int>(1) ; // dummy array providing somewhere to hang the SPECS
    u4r->fill(0) ; 
    u4r->set_names(U4Recorder::SPECS.names); 
    u4r->set_meta<std::string>("DescFakes", descfakes );  

    return u4r ; 
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

    spho ulabel = {} ; 
    GetLabel( ulabel, track ); 
    if(!Enabled(ulabel)) return ; // EIDX, GIDX skipping  

    if(is_fStopAndKill)
    {
        SEvt* sev = SEvt::Get_ECPU(); 
        U4Random::SetSequenceIndex(-1); 
        sev->finalPhoton(ulabel);  
        transient_fSuspend_track = nullptr ;
#ifndef PRODUCTION
        bool PIDX_DUMP = ulabel.id == PIDX && PIDX_ENABLED ; 
        sseq& seq = sev->current_ctx.seq ; 

        LOG_IF(info, PIDX_DUMP )    // CURIOUS : THIS IS NOT SHOWING UP 
            << " l.id " << std::setw(5) << ulabel.id
            << " seq " << seq.brief()
            ;  

        if(PIDX_DUMP) std::cerr
            << "U4Recorder::PostUserTrackingAction_Optical.fStopAndKill "
            << " ulabel.id " << std::setw(6) << ulabel.id 
            << " seq.brief " << seq.brief() 
            << std::endl
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

* YEP: this is happening when first post is on a fake 

**bop info is mostly missing**

*bop* was formerly only available WITH_PMTFASTSIM whilst using InstrumentedG4OpBoundaryProcess
as that ISA SOpBoundaryProcess giving access via SOpBoundaryProcess::INSTANCE 
have now generalize that to work with CustomG4OpBoundaryProcess

**Track Labelling** 

Q: Where is the STrackInfo labelling added to the track with FastSim ? 
A: Labelling added to track at the tail of the FastSim DoIt, eg "jcv junoPMTOpticalModel" 

**Limited Applicability Quantities**

1. For most step points the customBoundaryStatus is not applicable 
   it only applies to very specfic surfaces. 
   So getting it for every step point is kinda confusing.
   Need to scrub it when it doesnt apply, and cannot
   do that using is_boundary_flag. 
   How to detect when it is relevant from here ? 

2. Similarly when not at boundary the recoveredNormal is meaningless


**How to skip same material fakes from the FastSim-compromised-kludged-unnatural geometry ?**

::

          Vacuum | Vacuum
                 |
         -------+|+-------
                 |


**current_aux**

::

   current_aux.q1.i.w  : ascii status integer 'F' for first


**How to incorporate info from ProcessHits into the SEvt ?**

g4-cls G4SteppingManager::

    230 // Send G4Step information to Hit/Dig if the volume is sensitive
    231    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    232    StepControlFlag =  fStep->GetControlFlag();
    233    if( fCurrentVolume != 0 && StepControlFlag != AvoidHitInvocation) {
    234       fSensitive = fStep->GetPreStepPoint()->
    235                                    GetSensitiveDetector();
    236       if( fSensitive != 0 ) {
    237         fSensitive->Hit(fStep);
    238       }
    239    }
    240 
    241 // User intervention process.
    242    if( fUserSteppingAction != 0 ) {
    243       fUserSteppingAction->UserSteppingAction(fStep);
    244    }
    
* G4VSensitive::Hit/G4VSensitive::ProcessHits happens before UserSteppingAction
  so if plant the ProcessHits enum into the track label
  can then copy that into the current_aux  


Q: Where does the "TO" flag come from ?
A: See SEvt::beginPhoton
    
Q: Can the dependency on boundary process type be avoided ? 
A: HMM: Perhaps if CustomG4OpBoundary process inherited from G4OpBoundaryProcess
   avoid some complexities but maybe just add others. Would rather not go there. 
   Prefer simplicity. 


FIXED : Logging issue with U4 
------------------------------

Somehow SLOG-logging from U4 had stopped appearing. 
Turned out this was caused by the OPTICKS_U4 switch 
being accidentally flipped to PRIVATE in CMakeLists.txt
: that must stay PUBLIC for SLOG to work.  

**/

template <typename T>
void U4Recorder::UserSteppingAction_Optical(const G4Step* step)
{
    const G4Track* track = step->GetTrack(); 
    G4VPhysicalVolume* pv = track->GetVolume() ; 
    const G4VTouchable* touch = track->GetTouchable();  

    LOG(LEVEL) << "[  pv "  << ( pv ? pv->GetName() : "-" ) ; 

    spho ulabel = {} ; 
    GetLabel( ulabel, track ); 
    if(!Enabled(ulabel)) return ;   // EIDX, GIDX skipping 

    const G4StepPoint* pre = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    G4ThreeVector delta = step->GetDeltaPosition(); 
    double step_mm = delta.mag()/mm  ;   

    SEvt* sev = SEvt::Get_ECPU(); 
    sev->checkPhotonLineage(ulabel); 

    sphoton& current_photon = sev->current_ctx.p ;
    quad4&   current_aux    = sev->current_ctx.aux ; 
    current_aux.zero_v(3, 3);   // may be set below

    // first_flag identified by the flagmask having a single bit (all genflag are single bits, set in beginPhoton)
    bool first_flag = current_photon.flagmask_count() == 1 ;  
    if(first_flag)
    { 
        LOG(LEVEL) << " first_flag, track " << track ; 
        U4StepPoint::Update(current_photon, pre);   // populate current_photon with pos,mom,pol,time,wavelength
        current_aux.q1.i.w = int('F') ; 
        sev->pointPhoton(ulabel);        // sctx::point copying current into buffers 
    }
    unsigned flag = U4StepPoint::Flag<T>(post) ; 

    bool is_fastsim_flag = flag == DEFER_FSTRACKINFO ; 
    bool is_boundary_flag = OpticksPhoton::IsBoundaryFlag(flag) ;  // SD SA DR SR BR BT 
    bool is_surface_flag = OpticksPhoton::IsSurfaceDetectOrAbsorbFlag(flag) ;  // SD SA
    bool is_detect_flag = OpticksPhoton::IsSurfaceDetectFlag(flag) ;  // SD 
    if(is_boundary_flag) CollectBoundaryAux<T>(&current_aux) ;  

/*
#ifdef U4RECORDER_EXPENSIVE_IINDEX
    // doing replica number search for every step is very expensive and often pointless
    // its the kind of thing to do only for low stats or simple geometry running 
    current_photon.iindex = U4Touchable::ReplicaNumber(touch, REPLICA_NAME_SELECT);  
#else
    current_photon.iindex = is_surface_flag ? U4Touchable::ReplicaNumber(touch, REPLICA_NAME_SELECT) : -2 ;  
#endif
*/
    current_photon.iindex = is_detect_flag ? 
              U4Touchable::ImmediateReplicaNumber(touch) 
              :  
              U4Touchable::AncestorReplicaNumber(touch) 
              ;  

    LOG(LEVEL)
        << " flag " << flag
        << " " << OpticksPhoton::Flag(flag)
        << " is_fastsim_flag " << is_fastsim_flag  
        << " is_boundary_flag " << is_boundary_flag  
        << " is_surface_flag " << is_surface_flag  
        ;

    // DEFER_FSTRACKINFO : special flag signalling that 
    // the FastSim DoIt status needs to be accessed via the 
    // trackinfo label 
    //
    // FastSim status char "?DART" is set at the tail of junoPMTOpticalModel::DoIt

    if(is_fastsim_flag)
    {
        char fstrackinfo_stat = ulabel.uc4.w ; 
        // label->uc4.w = '_' ;  
        // scrub after access : HMM IS THIS NEEDED ? NOT EASY NOW THAN USE COPY: ulabel

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
            << " fstrackinfo_stat == '\0' " << ( fstrackinfo_stat == '\0' ? "YES" : "NO " )
            ;

        LOG(LEVEL) 
            << " DEFER_FSTRACKINFO " 
            << " fstrackinfo_stat " << fstrackinfo_stat 
            << " flag " << OpticksPhoton::Flag(flag) 
            ; 
    }

    LOG_IF(error, flag == 0) 
        << " ERR flag zero : post " 
        << std::endl 
        << "U4StepPoint::DescPositionTime(post)"
        << std::endl 
        << U4StepPoint::DescPositionTime(post) 
        << std::endl 
        << "U4StepPoint::Desc<T>(post)"
        << std::endl 
        << U4StepPoint::Desc<T>(post) 
        ;

    assert( flag > 0 ); 

    bool PIDX_DUMP = ulabel.id == PIDX && PIDX_ENABLED ; 
    bool is_fake = false ; 
    unsigned fakemask = 0 ;
    double   fake_duration(-2.); 

    int st = -2 ;   

    if(FAKES_SKIP && ( flag == BOUNDARY_TRANSMIT || flag == BOUNDARY_REFLECT ) )
    {
       // fake detection is very expensive : only do when needed

        std::string spec_ = U4Step::Spec(step) ;  // ctrl-c sampling suspect slow
        const char* spec = spec_.c_str(); 
        fakemask = ClassifyFake(step, flag, spec, PIDX_DUMP, &fake_duration ) ; 
        is_fake = fakemask > 0 ; 
        st = ( is_fake ? -1 : 1 )*SPECS.add(spec, false ) ;   
    }


    if(PIDX_DUMP) 
    {
        std::cout 
            << "U4Recorder::UserSteppingAction_Optical" 
            << " PIDX " << PIDX 
            << " post " << U4StepPoint::DescPositionTime(post) 
            << " is_fastsim_flag " << is_fastsim_flag 
            << " FAKES_SKIP " << FAKES_SKIP 
            << " is_fake " << is_fake
            << " fakemask " << fakemask 
            << std::endl 
            ; 
        LOG(LEVEL) << U4StepPoint::DescPositionTime(post) ;  
    }


    //current_aux.q2.u.z = ulabel.uc4packed();  // CAUTION: stomping on cdbg.pmtid setting above
    //current_aux.q2.i.w = st ;                  // CAUTION: stomping on cdbg.spare setting above  

    current_aux.q2.i.z = fakemask ;              // CAUTION: stomping on cdbg.pmtid setting above  
    current_aux.q2.f.w = float(fake_duration) ;  // CAUTION: stomping on cdbg.spare setting above 





    bool slow_fake = fake_duration > SLOW_FAKE ; 

    LOG_IF(info, PIDX_DUMP || slow_fake  ) 
        << " l.id " << std::setw(3) << ulabel.id
        << " step_mm " << std::fixed << std::setw(10) << std::setprecision(4) << step_mm 
        << " abbrev " << OpticksPhoton::Abbrev(flag)
        << " st " << std::setw(3) << st 
        << " is_fake " << ( is_fake ? "YES" : "NO " )
        << " fakemask " << fakemask
        << " fake_duration " << fake_duration 
        << " slow_fake " << slow_fake
        << " SLOW_FAKE " << SLOW_FAKE
        << " U4Fake::Desc " << U4Fake::Desc(fakemask)
        ;

    if( flag == NAN_ABORT )
    {
        LOG(LEVEL) << " skip post saving for StepTooSmall ulabel.id " << ulabel.id  ;  
    }
    else if( FAKES_SKIP && is_fake  )
    { 
        LOG(LEVEL) << " FAKES_SKIP skip post identified as fake ulabel.id " << ulabel.id  ;  
    }
    else
    {
        G4TrackStatus tstat = track->GetTrackStatus(); 

        Check_TrackStatus_Flag(tstat, flag, "UserSteppingAction_Optical" ); 

        U4StepPoint::Update(current_photon, post); 

        current_photon.set_flag( flag );

        if(U4Step::CF) U4Step::MockOpticksBoundaryIdentity(current_photon, step, ulabel.id ); 

        sev->pointPhoton(ulabel);     // save SEvt::current_photon/rec/seq/prd into sevent 
    }

    if(UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft)
    {
        U4Process::ClearNumberOfInteractionLengthLeft(*track, *step); 
    }

    LOG(LEVEL) << "]" ; 
}


const double U4Recorder::SLOW_FAKE = ssys::getenvdouble("U4Recorder__SLOW_FAKE", 1e-2) ; 
const bool U4Recorder::UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft = ssys::getenvbool(UserSteppingAction_Optical_ClearNumberOfInteractionLengthLeft_) ; 


/**
U4Recorder::CollectBoundaryAux
-------------------------------

Templated use from U4Recorder::UserSteppingAction_Optical

**/

template <typename T>
void U4Recorder::CollectBoundaryAux(quad4* )  // static
{
    LOG(LEVEL) << "generic do nothing" ; 
}


/**
U4Recorder::CollectBoundaryAux
--------------------------------

CAN THIS BE MOVED ELSEWHERE TO SIMPLIFY DEPS ? 

Maybe move into one of::

    CustomG4OpBoundaryProcess
    CustomART
    CustomART_Debug

**/

#if defined(WITH_CUSTOM4)
template<>
void U4Recorder::CollectBoundaryAux<C4OpBoundaryProcess>(quad4* current_aux)
{
    C4OpBoundaryProcess* bop = U4OpBoundaryProcess::Get<C4OpBoundaryProcess>() ;  
    assert(bop) ; 
    assert(current_aux); 

    char customStatus = bop ? bop->m_custom_status : 'B' ; 
    C4CustomART* cart   = bop ? bop->m_custom_art : nullptr ; 
    const double* recoveredNormal =  bop ? (const double*)&(bop->theRecoveredNormal) : nullptr ;  

#ifdef C4_DEBUG
    C4CustomART_Debug* cdbg = cart ? &(cart->dbg) : nullptr ;  
#else
    C4CustomART_Debug* cdbg = nullptr ; 
#endif

    LOG(LEVEL) 
        << " bop " << ( bop ? "Y" : "N" ) 
        << " cart " << ( cart ? "Y" : "N" )
        << " cdbg " << ( cdbg ? "Y" : "N" )
        << " current_aux " << ( current_aux ? "Y" : "N" )
        << " bop.m_custom_status " << customStatus
        << " CustomStatus::Name " << CustomStatus::Name(customStatus) 
        ; 

    if(cdbg && customStatus == 'Y') current_aux->load( cdbg->data(), C4CustomART_Debug::N ) ;   
    current_aux->set_v(3, recoveredNormal, 3);   // nullptr are just ignored
    current_aux->q3.i.w = int(customStatus) ;    // moved from q1 to q3
}

#elif defined(WITH_PMTSIM) || defined(WITH_CUSTOM_BOUNDARY)

// THIS CODE IS TO BE DELETED ONCE THE ABOVE IS WORKING 

template<>
void U4Recorder::CollectBoundaryAux<CustomG4OpBoundaryProcess>(quad4* current_aux)
{
    CustomG4OpBoundaryProcess* bop = U4OpBoundaryProcess::Get<CustomG4OpBoundaryProcess>() ;  
    assert(bop) ; 

    char customStatus = bop ? bop->m_custom_status : 'B' ; 
    CustomART* cart   = bop ? bop->m_custom_art : nullptr ; 
    const double* recoveredNormal =  bop ? (const double*)&(bop->theRecoveredNormal) : nullptr ;  
    CustomART_Debug* cdbg = cart ? &(cart->dbg) : nullptr ;  

    LOG(LEVEL) 
        << " bop " << ( bop ? "Y" : "N" ) 
        << " cart " << ( cart ? "Y" : "N" )
        << " cdbg " << ( cdbg ? "Y" : "N" )
        << " current_aux " << ( current_aux ? "Y" : "N" )
        << " bop.m_custom_status " << customStatus
        << " CustomStatus::Name " << CustomStatus::Name(customStatus) 
        ; 

    assert( current_aux ); 
    if(cdbg && customStatus == 'Y') 
    {
        // much of the contents of CustomART,CustomART_Debug 
        // only meaningful after doIt call : hence require customStatus 'Y'

        current_aux->q0.f.x = cdbg->A ; 
        current_aux->q0.f.y = cdbg->R ; 
        current_aux->q0.f.z = cdbg->T ; 
        current_aux->q0.f.w = cdbg->_qe ; 

        current_aux->q1.f.x = cdbg->An ; 
        current_aux->q1.f.y = cdbg->Rn ; 
        current_aux->q1.f.z = cdbg->Tn ; 
        current_aux->q1.f.w = cdbg->escape_fac ;  // HMM: this stomps on  ascii status integer 

        current_aux->q2.f.x = cdbg->minus_cos_theta ;
        current_aux->q2.f.y = cdbg->wavelength_nm  ; 
        current_aux->q2.f.z = cdbg->pmtid ;       // HMM: q2.i.z maybe set to fakemask below
        current_aux->q2.f.w = -1. ;               // HMM: q2.i.w gets set to step spec index  
    }

    current_aux->set_v(3, recoveredNormal, 3);   // nullptr are just ignored
    current_aux->q3.i.w = int(customStatus) ;    // moved from q1 to q3
}
#endif





/**
U4Recorder::ClassifyFake
--------------------------

Think about stepping around geometry with back foot "pre" and front foot "post". 
As take the next step the former "post" becomes the "pre" of the next step.  

U4Recorder operates by setting the flag and collecting info regarding 
"post" points of each step (pre, post) pair. The "pre" point only gets 
collected for the first step.  

Consider fake skipping a Vac/Vac coincident border::

       
                              | |                        |
                              | |                        |
       0----------------------1-2------------------------3---
                              | |                        |
                              | |                        |
                            coincident
                            border between 
                            volumes

Without any fake skipping would have::

* 0->1 
* 1->2
* 2->3 

With fake skipping that becomes:

* 0->3 

Notice that there is some conflation over whether should classify fake steps or fake points. 
Handling fake points would be cleaner but the info of the other point might be useful, 
so leaving asis given that current incantation seems to work. 

+-----------------+---------------------------------------------------------------------------+
| enum (0x1 << n) | U4Recorder::ClassifyFake heuristics, all contribute to fakemask           |  
+=================+===========================================================================+
| FAKE_STEP_MM    | step length less than EPSILON thats not a reflection turnaround           |
+-----------------+---------------------------------------------------------------------------+
| FAKE_FDIST      | distance to body_phys volume in direction of photon is less than EPSILON  |
+-----------------+---------------------------------------------------------------------------+
| FAKE_SURFACE    | body_phys solid frame localPoint EInside is kSurface (powerful)           |
+-----------------+---------------------------------------------------------------------------+
| FAKE_MANUAL     | manual selection via spec label (not recommended anymore)                 |
+-----------------+---------------------------------------------------------------------------+
| FAKE_VV_INNER12 | U4Step::IsSameMaterialPVBorder Vacuum inner1_phys/inner2_phys             |
+-----------------+---------------------------------------------------------------------------+

**/

const double U4Recorder::EPSILON = 1e-4 ; 
const bool U4Recorder::ClassifyFake_FindPV_r = ssys::getenvbool("U4Recorder__ClassifyFake_FindPV_r" ); 
stimer* U4Recorder::TIMER = new stimer ; 

unsigned U4Recorder::ClassifyFake(const G4Step* step, unsigned flag, const char* spec, bool dump, double* duration )
{
    if(duration) TIMER->start() ; 

    unsigned fakemask = 0 ; 
    G4ThreeVector delta = step->GetDeltaPosition(); 
    double step_mm = delta.mag()/mm  ;   

    // these are cheap and easy fake detection 
    bool is_reflect_flag = OpticksPhoton::IsReflectFlag(flag); 
    bool is_small_step   = step_mm < EPSILON && is_reflect_flag == false ; 
    bool is_vacvac_inner1_inner2 = U4Step::IsSameMaterialPVBorder(step, "Vacuum", "inner1_phys", "inner2_phys") ; 

    if(is_small_step)            fakemask |= U4Fake::FAKE_STEP_MM ; 
    if(is_vacvac_inner1_inner2)  fakemask |= U4Fake::FAKE_VV_INNER12 ; 
    if(IsListedFake(spec))       fakemask |= U4Fake::FAKE_MANUAL ;  


    // the below are powerful, but expensive fake detection

    const char* fake_pv_name = "body_phys" ; 
    const G4Track* track = step->GetTrack(); 
    const G4VTouchable* touch = track->GetTouchable();  
    const G4VPhysicalVolume* pv = track->GetVolume() ; 
    const char* pv_name = pv->GetName().c_str(); 
    bool pv_name_proceed = 
         strstr(pv_name, "PMT")  != nullptr ||
         strstr(pv_name, "nnvt") != nullptr ||
         strstr(pv_name, "hama") != nullptr 
         ; 

    // finding volume by name in the touch stack is much quicker than the recursive U4Volume::FindPV
    const G4VPhysicalVolume* fpv = U4Touchable::FindPV(touch, fake_pv_name, U4Touchable::MATCH_END );  
    int maxdepth = 2 ; 

    if( fpv == nullptr && ClassifyFake_FindPV_r && pv_name_proceed ) 
    {
        fpv = U4Volume::FindPV( pv, fake_pv_name, sstr::MATCH_END, maxdepth ); 
    }

    //LOG_IF(info, fpv == nullptr ) 
    // this is happening a lot 
    if(dump && fpv == nullptr) std::cout
        << "U4Recorder::ClassifyFake"   
        << " fpv null "
        << " pv_name " << ( pv_name ? pv_name : "-" )
        << " pv_name_proceed " << ( pv_name_proceed ? "YES" : "NO ") 
        << " ClassifyFake_FindPV_r " << ( ClassifyFake_FindPV_r ? "YES" : "NO " )
        << U4Touchable::Desc(touch) 
        << U4Touchable::Brief(touch) 
        << std::endl 
        ; 

    G4LogicalVolume* flv = fpv ? fpv->GetLogicalVolume() : nullptr ; 
    G4VSolid* fso = flv ? flv->GetSolid() : nullptr ; 

    const G4AffineTransform& transform = touch->GetHistory()->GetTopTransform();
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    const G4ThreeVector& theGlobalPoint = post->GetPosition(); 
    const G4ThreeVector& theGlobalDirection = post->GetMomentumDirection() ; 

    G4ThreeVector theLocalPoint     = transform.TransformPoint(theGlobalPoint); 
    G4ThreeVector theLocalDirection = transform.TransformAxis(theGlobalDirection); 
    G4ThreeVector theLocalIntersect(0.,0.,0.)  ; 

    EInside fin = kOutside ; 
    G4double fdist = fso == nullptr ? kInfinity : ssolid::Distance_( fso, theLocalPoint, theLocalDirection, fin, &theLocalIntersect ) ; 


    if(fdist < EPSILON)    fakemask |= U4Fake::FAKE_FDIST ;  
    if(fin == kSurface)    fakemask |= U4Fake::FAKE_SURFACE ; 

    if(duration) *duration = TIMER->done() ; 

    //LOG_IF(info, dump) 
    if(dump) std::cout 
        << "U4Recorder::ClassifyFake"   
        << " fdist " << ( fdist == kInfinity ? -1. : fdist )  
        << " fin " << sgeomdefs::EInside_(fin)
        << " fso " << ( fso ? fso->GetName() : "-" )
        << " theLocalPoint " << theLocalPoint     
        << " theLocalDirection " << theLocalDirection     
        << " theLocalIntersect " << theLocalIntersect     
        << " fakemask " << fakemask
        << " desc " << U4Fake::Desc(fakemask)
        << " duration " << std::scientific << ( duration ? *duration : -1. ) 
        << std::endl 
        ; 

    return fakemask ; 
}


std::vector<std::string>* U4Recorder::FAKES      = ssys::getenv_vec<std::string>("U4Recorder__FAKES", "" );
bool                      U4Recorder::FAKES_SKIP = ssys::getenvbool(             "U4Recorder__FAKES_SKIP") ;  

bool U4Recorder::IsListedFake( const char* spec ){ return ssys::is_listed(FAKES, spec ) ; }

std::string U4Recorder::DescFakes() // static
{
    std::stringstream ss ; 
    ss << "U4Recorder::DescFakes  " << std::endl  
       << "U4Recorder::FAKES_SKIP " << ( FAKES_SKIP ? "YES" : "NO " ) << std::endl 
       << "U4Recorder::FAKES      " << ( FAKES      ? "YES" : "NO " ) << std::endl 
       << "FAKES.size             " << ( FAKES ? FAKES->size() : -1 ) << std::endl 
       ;

    if(FAKES) for(int i=0 ; i < int(FAKES->size()) ; i++) ss << (*FAKES)[i] << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
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

        //assert( is_live_flag );  SKIP ASSERT
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



#if defined(WITH_CUSTOM4)
#include "C4OpBoundaryProcess.hh"
template void U4Recorder::UserSteppingAction_Optical<C4OpBoundaryProcess>(const G4Step*) ; 
#elif defined(WITH_PMTSIM)
#include "CustomG4OpBoundaryProcess.hh"
template void U4Recorder::UserSteppingAction_Optical<CustomG4OpBoundaryProcess>(const G4Step*) ; 
#else
#include "InstrumentedG4OpBoundaryProcess.hh"
template void U4Recorder::UserSteppingAction_Optical<InstrumentedG4OpBoundaryProcess>(const G4Step*) ; 
#endif



