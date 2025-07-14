#include <sstream>
#include <cstring>
#include <csignal>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ssys.h"
#include "sseq.h"
#include "sstr.h"
#include "spath.h"
#include "sdirectory.h"
#include "salloc.h"
#include "scontext.h"
#include "sbuild.h"
#include "sphoton.h"

#include "SPath.hh"   // only SPath::Make to replace



#include "SEventConfig.hh"
#include "SRG.h"  // raygenmode
#include "SRM.h"  // runningmode
#include "SComp.h"
#include "OpticksPhoton.hh"

#include "SLOG.hh"

const plog::Severity SEventConfig::LEVEL = SLOG::EnvLevel("SEventConfig", "DEBUG") ;

int         SEventConfig::_IntegrationModeDefault = -1 ;
const char* SEventConfig::_EventModeDefault = Minimal ;  // previously was Default
const char* SEventConfig::_EventNameDefault = nullptr ;
const char* SEventConfig::_RunningModeDefault = "SRM_DEFAULT" ;
int         SEventConfig::_StartIndexDefault = 0 ;
int         SEventConfig::_NumEventDefault = 1 ;
const char* SEventConfig::_NumPhotonDefault = nullptr ;
const char* SEventConfig::_NumGenstepDefault = nullptr ;
int         SEventConfig::_EventSkipaheadDefault = 100000 ;  // APPROPRIATE SKIPAHEAD DEPENDS ON HOW MANY RANDOMS CONSUMED BY PHOTON SIMULATION
const char* SEventConfig::_G4StateSpecDefault = "1000:38" ;
const char* SEventConfig::_G4StateSpecNotes   = "38=2*17+4 is appropriate for MixMaxRng" ;
int         SEventConfig::_G4StateRerunDefault = -1 ;

const char* SEventConfig::_MaxBounceNotes = "NB bounce limit is now separate from the non-PRODUCTION record limit which is inherent from sseq.h sseq::SLOTS " ;
const char* SEventConfig::_MaxTimeNotes = "NB time limit(ns) can truncate simulation together with bounce limit, default timer limit is so high to be unlimited " ;


int   SEventConfig::_MaxBounceDefault = 31 ;  // was previously too small at 9
float SEventConfig::_MaxTimeDefault = 1.e27f ; // crazy high default (ns) effectively meaning no limit

int SEventConfig::_MaxRecordDefault = 0 ;
int SEventConfig::_MaxRecDefault = 0 ;
int SEventConfig::_MaxAuxDefault = 0 ;
int SEventConfig::_MaxSupDefault = 0 ;
int SEventConfig::_MaxSeqDefault = 0 ;
int SEventConfig::_MaxPrdDefault = 0 ;
int SEventConfig::_MaxTagDefault = 0 ;
int SEventConfig::_MaxFlatDefault = 0 ;

float SEventConfig::_MaxExtentDomainDefault = 1000.f ;  // mm  : domain compression used by *rec*
float SEventConfig::_MaxTimeDomainDefault = 10.f ; // ns

const char* SEventConfig::_OutFoldDefault = "$DefaultOutputDir" ;
const char* SEventConfig::_OutNameDefault = nullptr ;
const char* SEventConfig::_EventReldirDefault = "ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-no_opticks_event_name}" ; // coordinate with kEventName
const char* SEventConfig::_RGModeDefault = "simulate" ;
const char* SEventConfig::_HitMaskDefault = "SD" ;


#if defined(RNG_XORWOW)
const char* SEventConfig::_MaxCurandDefault = "M3" ;
const char* SEventConfig::_MaxSlotDefault = "M3" ;
const char* SEventConfig::_MaxGenstepDefault = "M3" ;
const char* SEventConfig::_MaxPhotonDefault = "M3" ;
const char* SEventConfig::_MaxSimtraceDefault = "M3" ;

#elif defined(RNG_PHILOX) || defined(RNG_PHILITEOX)
const char* SEventConfig::_MaxCurandDefault = "G1" ; // nominal 1-billion states, as Philox has no need for curandState loading
const char* SEventConfig::_MaxSlotDefault = "0" ;     // see SEventConfig::SetDevice : set according to VRAM
const char* SEventConfig::_MaxGenstepDefault = "M10" ;  // adhoc
const char* SEventConfig::_MaxPhotonDefault = "G1" ;
const char* SEventConfig::_MaxSimtraceDefault = "G1" ;
#endif



const char* SEventConfig::_GatherCompDefault = SComp::ALL_ ;
const char* SEventConfig::_SaveCompDefault = SComp::ALL_ ;

float SEventConfig::_PropagateEpsilonDefault = 0.05f ;
float SEventConfig::_PropagateEpsilon0Default = 0.05f ;
const char* SEventConfig::_PropagateEpsilon0MaskDefault = "TO,CK,SI,SC,RE" ;
float SEventConfig::_PropagateRefineDistanceDefault = 5000.f ;


const char* SEventConfig::_InputGenstepDefault = nullptr ;
const char* SEventConfig::_InputGenstepSelectionDefault = nullptr ;
const char* SEventConfig::_InputPhotonDefault = nullptr ;
const char* SEventConfig::_InputPhotonFrameDefault = nullptr ;
float       SEventConfig::_InputPhotonChangeTimeDefault = -1.f ;   // -ve time means leave ASIS

int         SEventConfig::_IntegrationMode = ssys::getenvint(kIntegrationMode, _IntegrationModeDefault );
const char* SEventConfig::_EventMode = ssys::getenvvar(kEventMode, _EventModeDefault );
const char* SEventConfig::_EventName  = ssys::getenvvar(kEventName,  _EventNameDefault );
const char* SEventConfig::_DeviceName  = nullptr ;
int         SEventConfig::_RunningMode = SRM::Type(ssys::getenvvar(kRunningMode, _RunningModeDefault));




int         SEventConfig::_StartIndex = ssys::getenvint(kStartIndex, _StartIndexDefault );
int         SEventConfig::_NumEvent = ssys::getenvint(kNumEvent, _NumEventDefault );

std::vector<int>* SEventConfig::_GetNumPhotonPerEvent()
{
    const char* spec = ssys::getenvvar(kNumPhoton,  _NumPhotonDefault );
    return sstr::ParseIntSpecList<int>( spec, ',' );
}
std::vector<int>* SEventConfig::_NumPhotonPerEvent = _GetNumPhotonPerEvent() ;


std::vector<int>* SEventConfig::_GetNumGenstepPerEvent()
{
    const char* spec = ssys::getenvvar(kNumGenstep,  _NumGenstepDefault );
    return sstr::ParseIntSpecList<int>( spec, ',' );
}
std::vector<int>* SEventConfig::_NumGenstepPerEvent = _GetNumGenstepPerEvent() ;


/**
SEventConfig::_GetNumPhoton
----------------------------

Used by some tests that define a sequence of photon counts as inputs::

    OPTICKS_NUM_PHOTON=M1,2,3,4 OPTICKS_NUM_EVENT=4 SEventConfigTest


Expected to define the below of envvars with equal numbers of entries::

   OPTICKS_NUM_PHOTON=M1,2,3,4
   OPTICKS_NUM_GENSTEP=1,1,1,1


**/
int SEventConfig::_GetNumPhoton(int idx)
{
    if(_NumPhotonPerEvent == nullptr) return 0 ;

    int nevt0 = NumEvent() ;
    int nevt1 = _NumPhotonPerEvent->size() ;
    bool match = nevt0 == nevt1 ;
    LOG_IF(fatal, !match)
        << " NumEvent MISMATCH BETWEEN "
        << std::endl
        << " nevt0:_NumEvent               " << nevt0 << "( from " << kNumEvent  << ":" << ( getenv(kNumEvent) ? getenv(kNumEvent) : "-" ) << ") "
        << " nevt1:_NumPhotonPerEvent.size " << nevt1 << "( from " << kNumPhoton << ":" << ( getenv(kNumPhoton) ? getenv(kNumPhoton) : "-" ) << ") "
        ;
    assert( match );
    if(idx < 0 ) idx += nevt0 ;   // allow -ve indices to count from the back
    if(idx >= nevt0) return 0 ;

    return (*_NumPhotonPerEvent)[idx] ;
}


int SEventConfig::_GetNumGenstep(int idx)
{
    if(_NumGenstepPerEvent == nullptr) return 0 ;

    int nevt0 = NumEvent() ;
    int nevt1 = _NumGenstepPerEvent->size() ;
    bool match = nevt0 == nevt1 ;
    LOG_IF(fatal, !match)
        << " NumEvent MISMATCH BETWEEN "
        << std::endl
        << " nevt0:_NumEvent                " << nevt0 << "( from " << kNumEvent   << ":" << ( getenv(kNumEvent) ? getenv(kNumEvent) : "-" ) << ") "
        << " nevt1:_NumGenstepPerEvent.size " << nevt1 << "( from " << kNumGenstep << ":" << ( getenv(kNumGenstep) ? getenv(kNumGenstep) : "-" ) << ") "
        ;
    assert( match );
    if(idx < 0 ) idx += nevt0 ;   // allow -ve indices to count from the back
    if(idx >= nevt0) return 0 ;

    return (*_NumGenstepPerEvent)[idx] ;
}


/**
SEventConfig::_GetNumEvent
--------------------------

::

    OPTICKS_NUM_EVENT=4 OPTICKS_NUM_PHOTON=M1:3 OPTICKS_NUM_GENSTEP=M1:3 SEventConfigTest

**/

int SEventConfig::_GetNumEvent()
{
    bool have_NumPhotonPerEvent = _NumPhotonPerEvent && _NumPhotonPerEvent->size() > 0 ;
    bool have_NumGenstepPerEvent = _NumGenstepPerEvent && _NumGenstepPerEvent->size() > 0 ;

    int numEvent_fromPhotonList  = have_NumPhotonPerEvent ? _NumPhotonPerEvent->size() : 0 ;
    int numEvent_fromGenstepList = have_NumGenstepPerEvent ? _NumGenstepPerEvent->size() : 0 ;
    int numEvent_fromList = std::max(numEvent_fromPhotonList, numEvent_fromGenstepList);

    if( numEvent_fromPhotonList > 0 && numEvent_fromGenstepList > 0 )
    {
        assert( numEvent_fromPhotonList == numEvent_fromGenstepList );
    }

    bool override_NumEvent = numEvent_fromList > 0  && numEvent_fromList != _NumEvent ;

    LOG_IF(error, override_NumEvent )
        << " Overriding NumEvent "
        << "(" << kNumEvent  << ")"
        << " value " << _NumEvent
        << " as inconsistent with NumPhoton OR NumGenstep list length "
        << "\n"
        << "(" << kNumPhoton << ")"
        << " numEvent_fromPhotonList " << numEvent_fromPhotonList
        << "\n"
        << "(" << kNumGenstep << ")"
        << " numEvent_fromGenstepList " << numEvent_fromGenstepList
        ;
    return override_NumEvent ? numEvent_fromList : _NumEvent ;
}


int SEventConfig::NumPhoton( int idx){ return _GetNumPhoton(idx) ; }
int SEventConfig::NumGenstep(int idx){ return _GetNumGenstep(idx) ; }
int SEventConfig::NumEvent(){         return _GetNumEvent() ; }
int SEventConfig::EventIndex(int idx){ return _StartIndex + idx ; }
int SEventConfig::EventIndexArg(int index){ return index == MISSING_INDEX ? -1 : index - _StartIndex ; }

bool SEventConfig::IsFirstEvent(int idx){ return idx == 0 ; }  // 0-based idx (such as Geant4 eventID)
bool SEventConfig::IsLastEvent(int idx){ return idx == NumEvent()-1 ; }  // 0-based idx (such as Geant4 eventID)


int         SEventConfig::_EventSkipahead = ssys::getenvint(kEventSkipahead, _EventSkipaheadDefault) ;
const char* SEventConfig::_G4StateSpec  = ssys::getenvvar(kG4StateSpec,  _G4StateSpecDefault );
int         SEventConfig::_G4StateRerun = ssys::getenvint(kG4StateRerun, _G4StateRerunDefault) ;

int SEventConfig::_MaxCurand    = ssys::getenv_ParseInt(kMaxCurand,   _MaxCurandDefault ) ;
int SEventConfig::_MaxSlot      = ssys::getenv_ParseInt(kMaxSlot,     _MaxSlotDefault ) ;
int SEventConfig::_MaxGenstep   = ssys::getenv_ParseInt(kMaxGenstep,  _MaxGenstepDefault ) ;
int SEventConfig::_MaxPhoton    = ssys::getenv_ParseInt(kMaxPhoton,   _MaxPhotonDefault ) ;
int SEventConfig::_MaxSimtrace  = ssys::getenv_ParseInt(kMaxSimtrace, _MaxSimtraceDefault ) ;

int   SEventConfig::_MaxBounce    = ssys::getenvint(kMaxBounce, _MaxBounceDefault ) ;
float SEventConfig::_MaxTime    = ssys::getenvfloat(kMaxTime, _MaxTimeDefault );    // ns

int SEventConfig::_MaxRecord    = ssys::getenvint(kMaxRecord, _MaxRecordDefault ) ;
int SEventConfig::_MaxRec       = ssys::getenvint(kMaxRec, _MaxRecDefault ) ;
int SEventConfig::_MaxAux       = ssys::getenvint(kMaxAux, _MaxAuxDefault ) ;
int SEventConfig::_MaxSup       = ssys::getenvint(kMaxSup, _MaxSupDefault ) ;
int SEventConfig::_MaxSeq       = ssys::getenvint(kMaxSeq,  _MaxSeqDefault ) ;
int SEventConfig::_MaxPrd       = ssys::getenvint(kMaxPrd,  _MaxPrdDefault ) ;
int SEventConfig::_MaxTag       = ssys::getenvint(kMaxTag,  _MaxTagDefault ) ;
int SEventConfig::_MaxFlat      = ssys::getenvint(kMaxFlat,  _MaxFlatDefault ) ;

float SEventConfig::_MaxExtentDomain  = ssys::getenvfloat(kMaxExtentDomain, _MaxExtentDomainDefault );
float SEventConfig::_MaxTimeDomain    = ssys::getenvfloat(kMaxTimeDomain,   _MaxTimeDomainDefault );    // ns

const char* SEventConfig::_OutFold = ssys::getenvvar(kOutFold, _OutFoldDefault );
const char* SEventConfig::_OutName = ssys::getenvvar(kOutName, _OutNameDefault );
const char* SEventConfig::_EventReldir = ssys::getenvvar(kEventReldir, _EventReldirDefault );
int SEventConfig::_RGMode = SRG::Type(ssys::getenvvar(kRGMode, _RGModeDefault)) ;
unsigned SEventConfig::_HitMask  = OpticksPhoton::GetFlagMask(ssys::getenvvar(kHitMask, _HitMaskDefault )) ;

unsigned SEventConfig::_GatherComp  = SComp::Mask(ssys::getenvvar(kGatherComp, _GatherCompDefault )) ;
unsigned SEventConfig::_SaveComp    = SComp::Mask(ssys::getenvvar(kSaveComp,   _SaveCompDefault )) ;


float SEventConfig::_PropagateEpsilon = ssys::getenvfloat(kPropagateEpsilon, _PropagateEpsilonDefault ) ;
float SEventConfig::_PropagateEpsilon0 = ssys::getenvfloat(kPropagateEpsilon0, _PropagateEpsilon0Default ) ;
unsigned SEventConfig::_PropagateEpsilon0Mask  = OpticksPhoton::GetFlagMask(ssys::getenvvar(kPropagateEpsilon0Mask, _PropagateEpsilon0MaskDefault )) ;
std::string SEventConfig::PropagateEpsilon0MaskLabel(){  return OpticksPhoton::FlagMaskLabel( _PropagateEpsilon0Mask ) ; }
float SEventConfig::_PropagateRefineDistance = ssys::getenvfloat(kPropagateRefineDistance, _PropagateRefineDistanceDefault ) ;

const char* SEventConfig::_InputGenstep = ssys::getenvvar(kInputGenstep, _InputGenstepDefault );
const char* SEventConfig::_InputGenstepSelection = ssys::getenvvar(kInputGenstepSelection, _InputGenstepSelectionDefault );
const char* SEventConfig::_InputPhoton = ssys::getenvvar(kInputPhoton, _InputPhotonDefault );
const char* SEventConfig::_InputPhotonFrame = ssys::getenvvar(kInputPhotonFrame, _InputPhotonFrameDefault );
float SEventConfig::_InputPhotonChangeTime = ssys::getenvfloat(kInputPhotonChangeTime, _InputPhotonChangeTimeDefault ) ;


int         SEventConfig::IntegrationMode(){ return _IntegrationMode ; }
bool        SEventConfig::GPU_Simulation(){  return _IntegrationMode == 1 || _IntegrationMode == 3 ; }
bool        SEventConfig::CPU_Simulation(){  return _IntegrationMode == 2 || _IntegrationMode == 3 ; }

const char* SEventConfig::EventMode(){ return _EventMode ; }
const char* SEventConfig::EventName(){ return _EventName ; }
const char* SEventConfig::DeviceName(){ return _DeviceName ; }
bool        SEventConfig::HasDevice(){ return _DeviceName != nullptr ; }


/**
SEventConfig::RunningMode controlled via envvar OPTICKS_RUNNING_MODE
----------------------------------------------------------------------

* SRM_DEFAULT
* SRM_TORCH
* SRM_INPUT_PHOTON
* SRM_INPUT_GENSTEP
* SRM_GUN


**/


int         SEventConfig::RunningMode(){ return _RunningMode ; }
const char* SEventConfig::RunningModeLabel(){ return SRM::Name(_RunningMode) ; }

bool SEventConfig::IsRunningModeDefault(){      return RunningMode() == SRM_DEFAULT ; }
bool SEventConfig::IsRunningModeG4StateSave(){  return RunningMode() == SRM_G4STATE_SAVE ; }
bool SEventConfig::IsRunningModeG4StateRerun(){ return RunningMode() == SRM_G4STATE_RERUN ; }

bool SEventConfig::IsRunningModeTorch(){         return RunningMode() == SRM_TORCH ; }
bool SEventConfig::IsRunningModeInputPhoton(){   return RunningMode() == SRM_INPUT_PHOTON ; }
bool SEventConfig::IsRunningModeInputGenstep(){  return RunningMode() == SRM_INPUT_GENSTEP ; }
bool SEventConfig::IsRunningModeGun(){           return RunningMode() == SRM_GUN ; }


int SEventConfig::EventSkipahead(){ return _EventSkipahead ; }
const char* SEventConfig::G4StateSpec(){  return _G4StateSpec ; }

/**
SEventConfig::G4StateRerun
----------------------------

When rerun mode is not enabled returns -1 even when rerun id is set.

For a single photon rerun example see u4/tests/U4SimulateTest.cc
which uses U4Recorder::saveOrLoadStates from U4Recorder::PreUserTrackingAction_Optical

**/
int SEventConfig::G4StateRerun()
{
    bool rerun_enabled = IsRunningModeG4StateRerun() ;
    return rerun_enabled && _G4StateRerun > -1 ? _G4StateRerun : -1  ;
}



int SEventConfig::MaxCurand(){ return _MaxCurand ; }
int SEventConfig::MaxSlot(){   return _MaxSlot ; }

int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
int SEventConfig::MaxSimtrace(){   return _MaxSimtrace ; }


int SEventConfig::MaxBounce(){   return _MaxBounce ; }
float SEventConfig::MaxTime(){   return _MaxTime ; }

int SEventConfig::MaxRecord(){   return _MaxRecord ; }
int SEventConfig::MaxRec(){      return _MaxRec ; }
int SEventConfig::MaxAux(){      return _MaxAux ; }
int SEventConfig::MaxSup(){      return _MaxSup ; }
int SEventConfig::MaxSeq(){      return _MaxSeq ; }
int SEventConfig::MaxPrd(){      return _MaxPrd ; }
int SEventConfig::MaxTag(){      return _MaxTag ; }
int SEventConfig::MaxFlat(){      return _MaxFlat ; }

float SEventConfig::MaxExtentDomain(){ return _MaxExtentDomain ; }
float SEventConfig::MaxTimeDomain(){   return _MaxTimeDomain ; }

const char* SEventConfig::OutFold(){   return _OutFold ; }
const char* SEventConfig::OutName(){   return _OutName ; }

/**
SEventConfig::EventReldir
---------------------------

Usually left at the default of::

     ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-no_opticks_event_name}

Note that if OPTICKS_EVENT_NAME is defined it is constrained by
SEventConfig::Initialize_EventName to contain the build context string,
eg Debug_Philox.

**/
const char* SEventConfig::EventReldir(){   return _EventReldir ; }
unsigned SEventConfig::HitMask(){     return _HitMask ; }

unsigned SEventConfig::GatherComp(){  return _GatherComp ; }
unsigned SEventConfig::SaveComp(){    return _SaveComp ; }


float SEventConfig::PropagateEpsilon(){ return _PropagateEpsilon ; }
float SEventConfig::PropagateEpsilon0(){ return _PropagateEpsilon0 ; }
unsigned SEventConfig::PropagateEpsilon0Mask(){ return _PropagateEpsilon0Mask ; }
float SEventConfig::PropagateRefineDistance(){ return _PropagateRefineDistance ; }


/**
SEventConfig::_InputGenstepPath
--------------------------------

OPTICKS_INPUT_GENSTEP
    must provide a path with %d format tokens that are filled with idx

**/

const char* SEventConfig::_InputGenstepPath(int idx)
{
    return sstr::Format(_InputGenstep, idx ) ;
}
const char* SEventConfig::InputGenstep(int idx)
{
    return ( idx == -1 || _InputGenstep == nullptr ) ? _InputGenstep : _InputGenstepPath(idx) ;
}
const char* SEventConfig::InputGenstepSelection(int /*idx*/) // for now one selection for all eventID
{
    return _InputGenstepSelection  ;
}






bool SEventConfig::InputGenstepPathExists(int idx)
{
    const char* path = SEventConfig::InputGenstep(idx);
    return path ? spath::Exists(path) : false ;
}



/**
SEventConfig::InputPhoton control via OPTICKS_INPUT_PHOTON envvar
------------------------------------------------------------------

Pick the array of input_photons to use,
default none, eg "RainXZ_Z230_10k_f8.npy"

* when configured SEvt::initInputPhoton loads the input photons array
* within Opticks the photons are uploaded to the GPU with QEvent::setInputPhoton

Techniques to get the same input photons into Geant4 simulations, eg for A-B comparisons
between Opticks and Geant4 depend on the level of access to G4Event that is
afforded by the simulation framework.

* JUNOSW/GtOpticksTool "GenTool" uses a mutate interface to inject the photons via HepMC

* G4CXApp.h used from the raindrop example uses are more direct approach with
  U4VPrimaryGenerator::GeneratePrimaries_From_Photons using direct access to G4Event


When input photons are configured and found the accessor SEvt::hasInputPhoton
returns true which is consulted by SEvt::addInputGenstep resulting in creation
of the input photon genstep via SEvent::MakeInputPhotonGenstep.

Adding this fabricated input photon genstep kicks off the configured allocations.

**/

const char* SEventConfig::InputPhoton(){   return _InputPhoton ; }


/**
SEventConfig::InputPhotonFrame control via OPTICKS_INPUT_PHOTON_FRAME envvar
------------------------------------------------------------------------------

Pick the frame in which to inject the input photons,
using MOI style specification eg "NNVT:0:1000"


**/
const char* SEventConfig::InputPhotonFrame(){       return _InputPhotonFrame ; }
float       SEventConfig::InputPhotonChangeTime(){  return _InputPhotonChangeTime ; }


int SEventConfig::RGMode(){  return _RGMode ; }
bool SEventConfig::IsRGModeRender(){   return RGMode() == SRG_RENDER   ; }
bool SEventConfig::IsRGModeSimtrace(){ return RGMode() == SRG_SIMTRACE ; }
bool SEventConfig::IsRGModeSimulate(){ return RGMode() == SRG_SIMULATE ; }
bool SEventConfig::IsRGModeTest(){     return RGMode() == SRG_TEST ; }
const char* SEventConfig::RGModeLabel(){ return SRG::Name(_RGMode) ; }




void SEventConfig::SetDebugHeavy(){         SetEventMode(DebugHeavy) ; }
void SEventConfig::SetDebugLite(){          SetEventMode(DebugLite) ; }
void SEventConfig::SetNothing(){            SetEventMode(Nothing)           ; }
void SEventConfig::SetMinimal(){            SetEventMode(Minimal)           ; }
void SEventConfig::SetHit(){                SetEventMode(Hit)               ; }
void SEventConfig::SetHitPhoton(){          SetEventMode(HitPhoton)         ; }
void SEventConfig::SetHitPhotonSeq(){       SetEventMode(HitPhotonSeq)      ; }
void SEventConfig::SetHitSeq(){             SetEventMode(HitSeq)            ; }

bool SEventConfig::IsDebugHeavy(){        return _EventMode && strcmp(_EventMode, DebugHeavy) == 0 ; }
bool SEventConfig::IsDebugLite(){         return _EventMode && strcmp(_EventMode, DebugLite) == 0 ; }
bool SEventConfig::IsNothing(){           return _EventMode && strcmp(_EventMode, Nothing) == 0 ; }
bool SEventConfig::IsMinimal(){           return _EventMode && strcmp(_EventMode, Minimal) == 0 ; }
bool SEventConfig::IsHit(){               return _EventMode && strcmp(_EventMode, Hit) == 0 ; }
bool SEventConfig::IsHitPhoton(){         return _EventMode && strcmp(_EventMode, HitPhoton) == 0 ; }
bool SEventConfig::IsHitPhotonSeq(){      return _EventMode && strcmp(_EventMode, HitPhotonSeq) == 0 ; }
bool SEventConfig::IsHitSeq(){            return _EventMode && strcmp(_EventMode, HitSeq) == 0 ; }

bool SEventConfig::IsMinimalOrNothing(){ return IsMinimal() ||  IsNothing() ; }


std::string SEventConfig::DescEventMode()  // static
{
    std::stringstream ss ;
    ss << "SEventConfig::DescEventMode" << std::endl
       << DebugHeavy
       << std::endl
       << DebugLite
       << std::endl
       << Nothing
       << std::endl
       << Minimal
       << std::endl
       << Hit
       << std::endl
       << HitPhoton
       << std::endl
       << HitPhotonSeq
       << std::endl
       << HitSeq
       << std::endl
       ;

    std::string str = ss.str() ;
    return str ;
}


void SEventConfig::SetIntegrationMode(int mode){ _IntegrationMode = mode ; LIMIT_Check() ; }
void SEventConfig::SetEventMode(const char* mode){ _EventMode = mode ? strdup(mode) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetEventName(const char* name){ _EventName = name ? strdup(name) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetRunningMode(const char* mode){ _RunningMode = SRM::Type(mode) ; LIMIT_Check() ; }

void SEventConfig::SetStartIndex(int index0){        _StartIndex = index0 ; LIMIT_Check() ; }
void SEventConfig::SetNumEvent(int nevt){            _NumEvent = nevt ; LIMIT_Check() ; }
void SEventConfig::SetG4StateSpec(const char* spec){ _G4StateSpec = spec ? strdup(spec) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetG4StateRerun(int id){          _G4StateRerun = id ; LIMIT_Check() ; }

void SEventConfig::SetMaxCurand(int max_curand){ _MaxCurand = max_curand ; LIMIT_Check() ; }
void SEventConfig::SetMaxSlot(int max_slot){     _MaxSlot    = max_slot  ; LIMIT_Check() ; }

void SEventConfig::SetMaxGenstep(int max_genstep){ _MaxGenstep = max_genstep ; LIMIT_Check() ; }
void SEventConfig::SetMaxPhoton( int max_photon){  _MaxPhoton  = max_photon  ; LIMIT_Check() ; }
void SEventConfig::SetMaxSimtrace( int max_simtrace){  _MaxSimtrace  = max_simtrace  ; LIMIT_Check() ; }

void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; LIMIT_Check() ; }
void SEventConfig::SetMaxTime(   float max_time){   _MaxTime   = max_time  ; LIMIT_Check() ; }

void SEventConfig::SetMaxRecord( int max_record){  _MaxRecord  = max_record  ; LIMIT_Check() ; }
void SEventConfig::SetMaxRec(    int max_rec){     _MaxRec     = max_rec     ; LIMIT_Check() ; }
void SEventConfig::SetMaxAux(    int max_aux){     _MaxAux     = max_aux     ; LIMIT_Check() ; }
void SEventConfig::SetMaxSup(    int max_sup){     _MaxSup     = max_sup     ; LIMIT_Check() ; }
void SEventConfig::SetMaxSeq(    int max_seq){     _MaxSeq     = max_seq     ; LIMIT_Check() ; }
void SEventConfig::SetMaxPrd(    int max_prd){     _MaxPrd     = max_prd     ; LIMIT_Check() ; }
void SEventConfig::SetMaxTag(    int max_tag){     _MaxTag     = max_tag     ; LIMIT_Check() ; }
void SEventConfig::SetMaxFlat(    int max_flat){     _MaxFlat     = max_flat     ; LIMIT_Check() ; }

void SEventConfig::SetMaxExtentDomain( float max_extent){ _MaxExtentDomain = max_extent  ; LIMIT_Check() ; }
void SEventConfig::SetMaxTimeDomain(   float max_time){   _MaxTimeDomain = max_time  ; LIMIT_Check() ; }


void SEventConfig::SetOutFold(   const char* outfold){   _OutFold = outfold ? strdup(outfold) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetOutName(   const char* outname){   _OutName = outname ? strdup(outname) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetEventReldir(   const char* v){   _EventReldir = v ? strdup(v) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetHitMask(   const char* abrseq, char delim){  _HitMask = OpticksPhoton::GetFlagMask(abrseq,delim) ; }

void SEventConfig::SetRGModeSimulate(){  SetRGMode( SRG::SIMULATE_ ); }
void SEventConfig::SetRGModeSimtrace(){  SetRGMode( SRG::SIMTRACE_ ); }
void SEventConfig::SetRGModeRender(){    SetRGMode( SRG::RENDER_ ); }
void SEventConfig::SetRGModeTest(){      SetRGMode( SRG::TEST_ ); }

void SEventConfig::SetRGMode( const char* mode)
{
    int prior_RGMode = _RGMode ;
    _RGMode = SRG::Type(mode) ;
    bool changed_mode = prior_RGMode != _RGMode ;
    if(changed_mode)
    {
        LOG(LEVEL) << " mode changed calling Initialize_Comp " ;
        Initialize_Comp();
        LOG(LEVEL) << " DescGatherComp " << DescGatherComp() ;
        LOG(LEVEL) << " DescSaveComp   " << DescSaveComp() ;
    }
    LIMIT_Check() ;
}





void SEventConfig::SetPropagateEpsilon(float eps){ _PropagateEpsilon = eps ; LIMIT_Check() ; }
void SEventConfig::SetPropagateEpsilon0(float eps){ _PropagateEpsilon0 = eps ; LIMIT_Check() ; }
void SEventConfig::SetPropagateEpsilon0Mask(const char* abrseq, char delim){ _PropagateEpsilon0Mask = OpticksPhoton::GetFlagMask(abrseq,delim) ; }
void SEventConfig::SetPropagateRefineDistance(float refine_distance){ _PropagateRefineDistance = refine_distance ; LIMIT_Check() ; }

void SEventConfig::SetInputGenstep(const char* ig){   _InputGenstep = ig ? strdup(ig) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetInputGenstepSelection(const char* igsel){   _InputGenstepSelection = igsel ? strdup(igsel) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetInputPhoton(const char* ip){   _InputPhoton = ip ? strdup(ip) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetInputPhotonFrame(const char* ip){   _InputPhotonFrame = ip ? strdup(ip) : nullptr ; LIMIT_Check() ; }
void SEventConfig::SetInputPhotonChangeTime(float t0){    _InputPhotonChangeTime = t0 ; LIMIT_Check() ; }

void SEventConfig::SetGatherComp_(unsigned mask){ _GatherComp = mask ; }
void SEventConfig::SetGatherComp(const char* names, char delim){  SetGatherComp_( SComp::Mask(names,delim)) ; }
bool SEventConfig::GatherRecord(){  return ( _GatherComp & SCOMP_RECORD ) != 0 ; }

void SEventConfig::SetSaveComp_(unsigned mask){ _SaveComp = mask ; }
void SEventConfig::SetSaveComp(const char* names, char delim){  SetSaveComp_( SComp::Mask(names,delim)) ; }


//std::string SEventConfig::DescHitMask(){   return OpticksPhoton::FlagMaskLabel( _HitMask ) ; }
std::string SEventConfig::HitMaskLabel(){  return OpticksPhoton::FlagMaskLabel( _HitMask ) ; }

std::string SEventConfig::DescGatherComp(){ return SComp::Desc( _GatherComp ) ; }
std::string SEventConfig::DescSaveComp(){   return SComp::Desc( _SaveComp ) ; } // used from SEvt::save


void SEventConfig::GatherCompList( std::vector<unsigned>& gather_comp )
{
    SComp::CompListMask(gather_comp, GatherComp() );
}
int SEventConfig::NumGatherComp()
{
    return SComp::CompListCount(GatherComp() );
}

void SEventConfig::SaveCompList( std::vector<unsigned>& save_comp )
{
    SComp::CompListMask(save_comp, SaveComp() );
}
int SEventConfig::NumSaveComp()
{
    return SComp::CompListCount(SaveComp() );
}










/**
SEventConfig::LIMIT_Check
---------------------------

Since moved to compound stag/sflat in stag.h
MaxTag/MaxFlat must now either be 0 or 1, nothing else.
Had a bug previously with MaxTag/MaxFlat 24 that
caused huge memory allocations in debug event modes.

**/

int SEventConfig::RecordLimit() // static
{
    return sseq::SLOTS ;
}

void SEventConfig::LIMIT_Check()
{
   assert( _IntegrationMode >= -1 && _IntegrationMode <= 3 );

   //assert( _MaxBounce >= 0 && _MaxBounce <  LIMIT ) ;
   // MaxBounce should not in principal be limited

   assert( _MaxRecord >= 0 && _MaxRecord <= RecordLimit() ) ;
   assert( _MaxRec    >= 0 && _MaxRec    <= RecordLimit() ) ;
   assert( _MaxPrd    >= 0 && _MaxPrd    <= RecordLimit() ) ;

   assert( _MaxSeq    >= 0 && _MaxSeq    <= 1 ) ;    // formerly incorrectly allowed up to LIMIT
   assert( _MaxTag    >= 0 && _MaxTag    <= 1 ) ;
   assert( _MaxFlat   >= 0 && _MaxFlat   <= 1 ) ;

   assert( _StartIndex >= 0 );
}


std::string SEventConfig::Desc()
{
    std::stringstream ss ;
    ss << "SEventConfig::Desc" << std::endl
       << std::setw(25) << kIntegrationMode
       << std::setw(20) << " IntegrationMode " << " : " << IntegrationMode()
       << std::endl
       << std::setw(25) << kEventMode
       << std::setw(20) << " EventMode " << " : " << EventMode()
       << std::endl
       << std::setw(25) << kEventName
       << std::setw(20) << " EventName " << " : " << ( EventName() ? EventName() : "-" )
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " DeviceName " << " : " << ( DeviceName() ? DeviceName() : "-" )
       << std::endl
       << std::setw(25) << kRunningMode
       << std::setw(20) << " RunningMode " << " : " << RunningMode()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " RunningModeLabel " << " : " << RunningModeLabel()
       << std::endl
       << std::setw(25) << kNumEvent
       << std::setw(20) << " NumEvent " << " : " << NumEvent()
       << std::endl
       << std::setw(25) << kNumPhoton
       << std::setw(20) << " NumPhoton(0) " << " : " << NumPhoton(0)
       << std::setw(20) << " NumPhoton(1) " << " : " << NumPhoton(1)
       << std::setw(20) << " NumPhoton(-1) " << " : " << NumPhoton(-1)
       << std::endl
       << std::setw(25) << kNumGenstep
       << std::setw(20) << " NumGenstep(0) " << " : " << NumGenstep(0)
       << std::setw(20) << " NumGenstep(1) " << " : " << NumGenstep(1)
       << std::setw(20) << " NumGenstep(-1) " << " : " << NumGenstep(-1)
       << std::endl
       << std::setw(25) << kG4StateSpec
       << std::setw(20) << " G4StateSpec " << " : " << G4StateSpec()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " G4StateSpecNotes " << " : " << _G4StateSpecNotes
       << std::endl
       << std::setw(25) << kG4StateRerun
       << std::setw(20) << " G4StateRerun " << " : " << G4StateRerun()
       << std::endl
       << std::setw(25) << kMaxCurand
       << std::setw(20) << " MaxCurand " << " : " << MaxCurand()
       << std::setw(20) << " MaxCurand/M " << " : " << MaxCurand()/M
       << std::endl
       << std::setw(25) << kMaxSlot
       << std::setw(20) << " MaxSlot " << " : " << MaxSlot()
       << std::setw(20) << " MaxSlot/M " << " : " << MaxSlot()/M
       << std::endl
       << std::setw(25) << kMaxGenstep
       << std::setw(20) << " MaxGenstep " << " : " << MaxGenstep()
       << std::setw(20) << " MaxGenstep/M " << " : " << MaxGenstep()/M
       << std::endl
       << std::setw(25) << kMaxPhoton
       << std::setw(20) << " MaxPhoton " << " : " << MaxPhoton()
       << std::setw(20) << " MaxPhoton/M " << " : " << MaxPhoton()/M
       << std::endl
       << std::setw(25) << kMaxSimtrace
       << std::setw(20) << " MaxSimtrace " << " : " << MaxSimtrace()
       << std::setw(20) << " MaxSimtrace/M " << " : " << MaxSimtrace()/M
       << std::endl
       << std::setw(25) << kMaxBounce
       << std::setw(20) << " MaxBounce " << " : " << MaxBounce()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " MaxBounceNotes " << " : " << _MaxBounceNotes
       << std::endl
       << std::setw(25) << kMaxTime
       << std::setw(20) << " MaxTime " << " : " << MaxTime()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " MaxTimeNotes " << " : " << _MaxTimeNotes
       << std::endl
       << std::setw(25) << kMaxRecord
       << std::setw(20) << " MaxRecord " << " : " << MaxRecord()
       << std::endl
       << std::setw(25) << kMaxRec
       << std::setw(20) << " MaxRec " << " : " << MaxRec()
       << std::endl
       << std::setw(25) << kMaxAux
       << std::setw(20) << " MaxAux " << " : " << MaxAux()
       << std::endl
       << std::setw(25) << kMaxSup
       << std::setw(20) << " MaxSup " << " : " << MaxSup()
       << std::endl
       << std::setw(25) << kMaxSeq
       << std::setw(20) << " MaxSeq " << " : " << MaxSeq()
       << std::endl
       << std::setw(25) << kMaxPrd
       << std::setw(20) << " MaxPrd " << " : " << MaxPrd()
       << std::endl
       << std::setw(25) << kMaxTag
       << std::setw(20) << " MaxTag " << " : " << MaxTag()
       << std::endl
       << std::setw(25) << kMaxFlat
       << std::setw(20) << " MaxFlat " << " : " << MaxFlat()
       << std::endl
       << std::setw(25) << kHitMask
       << std::setw(20) << " HitMask " << " : " << HitMask()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " HitMaskLabel " << " : " << HitMaskLabel()
       << std::endl
       << std::setw(25) << kMaxExtentDomain
       << std::setw(20) << " MaxExtentDomain " << " : " << MaxExtentDomain()
       << std::endl
       << std::setw(25) << kMaxTimeDomain
       << std::setw(20) << " MaxTimeDomain " << " : " << MaxTimeDomain()
       << std::endl
       << std::setw(25) << kRGMode
       << std::setw(20) << " RGMode " << " : " << RGMode()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " RGModeLabel " << " : " << RGModeLabel()
       << std::endl
       << std::setw(25) << kGatherComp
       << std::setw(20) << " GatherComp " << " : " << GatherComp()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " DescGatherComp " << " : " << DescGatherComp()
       << std::endl
       << std::setw(25) << kSaveComp
       << std::setw(20) << " SaveComp " << " : " << SaveComp()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " DescSaveComp " << " : " << DescSaveComp()
       << std::endl
       << std::setw(25) << kOutFold
       << std::setw(20) << " OutFold " << " : " << OutFold()
       << std::endl
       << std::setw(25) << kOutName
       << std::setw(20) << " OutName " << " : " << ( OutName() ? OutName() : "-" )
       << std::endl
       << std::setw(25) << kEventReldir
       << std::setw(20) << " EventReldir " << " : " << ( EventReldir() ? EventReldir() : "-" )
       << std::endl
       << std::setw(25) << kPropagateEpsilon
       << std::setw(20) << " PropagateEpsilon " << " : " << std::fixed << std::setw(10) << std::setprecision(4) << PropagateEpsilon()
       << std::endl
       << std::setw(25) << kPropagateEpsilon0
       << std::setw(20) << " PropagateEpsilon0 " << " : " << std::fixed << std::setw(10) << std::setprecision(4) << PropagateEpsilon0()
       << std::endl
       << std::setw(25) << kPropagateEpsilon0Mask
       << std::setw(20) << " PropagateEpsilon0Mask " << " : " << PropagateEpsilon0Mask()
       << std::endl
       << std::setw(25) << ""
       << std::setw(20) << " PropagateEpsilon0MaskLabel " << " : " << PropagateEpsilon0MaskLabel()
       << std::endl
       << std::setw(25) << kPropagateRefineDistance
       << std::setw(20) << " PropagateRefineDistance " << " : " << PropagateRefineDistance()
       << std::endl
       << std::setw(25) << kInputGenstep
       << std::setw(20) << " InputGenstep " << " : " << ( InputGenstep() ? InputGenstep() : "-" )
       << std::endl
       << std::setw(25) << kInputGenstepSelection
       << std::setw(20) << " InputGenstepSelection " << " : " << ( InputGenstepSelection() ? InputGenstepSelection() : "-" )
       << std::endl
       << std::setw(25) << kInputPhoton
       << std::setw(20) << " InputPhoton " << " : " << ( InputPhoton() ? InputPhoton() : "-" )
       << std::endl
       << std::setw(25) << kInputPhotonChangeTime
       << std::setw(20) << " InputPhotonChangeTime " << " : " << InputPhotonChangeTime()
       << std::endl
       << std::setw(25) << "RecordLimit() "
       << std::setw(20) << " (sseq::SLOTS) " << " : " << RecordLimit()
       << std::endl
       ;
    std::string s = ss.str();
    return s ;
}


/**
SEventConfig::OutDir SEventConfig::OutPath
--------------------------------------------

Used by CSGOptiX::render_snap

Expecting the below as the OutName defaults to nullptr::

   $TMP/GEOM/$GEOM/ExecutableName

**/

const char* SEventConfig::OutDir()
{
    const char* outfold = OutFold();
    const char* outname = OutName();

    LOG(LEVEL)
        << " outfold " << ( outfold ? outfold : "-" )
        << " outname " << ( outname ? outname : "-" )
        ;

    const char* dir = outname == nullptr ?
                            spath::Resolve( outfold )
                            :
                            spath::Resolve( outfold, outname )
                            ;

    LOG(LEVEL)
        << " dir " << ( dir ? dir : "-" )
        ;

    sdirectory::MakeDirs(dir,0);
    return dir ;
}

/**
SEventConfig::OutPath
----------------------

unique:true
    when outpath file exists already increment the index until a non-existing outpath is found

**/

const char* SEventConfig::OutPath( const char* stem, int index, const char* ext, bool unique )
{
    const char* outfold = OutFold();
    const char* outname = OutName();

    LOG(LEVEL)
        << " outfold " << ( outfold ? outfold : "-" )
        << " outname " << ( outname ? outname : "-" )
        << " stem " << ( stem ? stem : "-" )
        << " ext " << ( ext ? ext : "-" )
        << " index " << index
        << " unique " << ( unique ? "Y" : "N" )
        ;

    const char* outpath = SPath::Make( outfold, outname, stem, index, ext, FILEPATH);

    if(unique)
    {
        // increment until find non-existing path
        int offset = 0 ;
        while( SPath::Exists(outpath) && offset < 100 )
        {
            offset += 1 ;
            outpath = SPath::Make( outfold, outname, stem, index+offset, ext, FILEPATH);
        }
    }


    return outpath ;
   // HMM: an InPath would use NOOP to not create the dir
}

const char* SEventConfig::OutPath( const char* reldir, const char* stem, int index, const char* ext, bool unique )
{
    const char* outfold = OutFold();
    const char* outname = OutName();
    LOG(LEVEL)
        << " outfold " << ( outfold ? outfold : "-" )
        << " outname " << ( outname ? outname : "-" )
        << " stem " << ( stem ? stem : "-" )
        << " ext " << ( ext ? ext : "-" )
        << " index " << index
        << " unique " << ( unique ? "Y" : "N" )
        ;

    const char* outpath = SPath::Make( outfold, outname, reldir, stem, index, ext, FILEPATH);

    if(unique)
    {
        // increment until find non-existing path
        int offset = 0 ;
        while( SPath::Exists(outpath) && offset < 100 )
        {
            offset += 1 ;
            outpath = SPath::Make( outfold, outname, reldir, stem, index+offset, ext, FILEPATH);
        }
    }
    return outpath ;
}


std::string SEventConfig::DescOutPath(  const char* stem, int index, const char* ext, bool unique)
{
    const char* path = OutPath(stem, index, ext, unique ) ;
    std::stringstream ss ;
    ss << "SEventConfig::DescOutPath" << std::endl
       << " stem " << ( stem ? stem : "-" )
       << " index " << index
       << " ext " << ( ext ? ext : "-" )
       << " unique " << ( unique ? "Y" : "N" )
       << std::endl
       << " OutFold " << OutFold()
       << " OutName " << OutName()
       << std::endl
       << " OutPath " <<  path
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}




const char* SEventConfig::OutDir(const char* reldir)
{
    const char* dir = spath::Resolve( OutFold(), OutName(), reldir );
    sdirectory::MakeDirs(dir, 0);
    return dir ;
}



scontext* SEventConfig::CONTEXT = nullptr ;
salloc*   SEventConfig::ALLOC = nullptr ;

std::string SEventConfig::GetGPUMeta(){ return CONTEXT ? CONTEXT->brief() : "ERR-NO-SEventConfig-CONTEXT" ; }


/**
SEventConfig::Initialize
-------------------------

Canonically invoked from SEvt::SEvt

* SO: must make any static call adjustments before SEvt instanciation


DebugHeavy
    far too heavy for most debugging/validation
DebugLite
    mode with photon, record, seq,  genstep covers most needs


Future
~~~~~~~

* maybe move home from SEvt to SSim, as only one SSim but can be two or more SEvt sometimes

**/

int SEventConfig::Initialize_COUNT = 0 ;
int SEventConfig::Initialize() // static
{
    LOG_IF(LEVEL, Initialize_COUNT > 0 )
        << "SEventConfig::Initialize() called more than once " << std::endl
        << " this is now done automatically at SEvt::SEvt usually from main "
        << " (IN SOME CASES ITS CONVENIENT TO HAVE MORE THAN ONE SEvt, THOUGH "
        << "  SO MAYBE SHOULD MOVE THIS TO OPTICKS_LOG/SLOG ? "
        ;

    if(Initialize_COUNT == 0)
    {
        Initialize_Meta() ;
        Initialize_EventName() ;
        Initialize_Comp() ;
    }
    Initialize_COUNT += 1 ;
    return 0 ;
}


void SEventConfig::Initialize_Meta()
{
    CONTEXT = new scontext ;
    ALLOC = new salloc ;
}

/**
SEventConfig::Initialize_EventName
-----------------------------------

Examples that would match some builds::

   export OPTICKS_EVENT_NAME="SomePrefix_Debug_Philox_SomeSuffix"
   export OPTICKS_EVENT_NAME="Debug_XORWOW"

**/

void SEventConfig::Initialize_EventName()
{
    if(EventName()==nullptr) return ;

    bool build_matches_EventName = sbuild::Matches(EventName()) ;

    LOG(LEVEL)
        << "\n"
        << " kEventName " << kEventName << "\n"
        << " SEventConfig::EventName() " << EventName() << "\n"
        << " build_matches_EventName " << ( build_matches_EventName ? "YES" : "NO " ) << "\n"
        << sbuild::Desc()
        ;

    LOG_IF( error, !build_matches_EventName)
        << "\n"
        << " kEventName " << kEventName << "\n"
        << " SEventConfig::EventName() " << EventName() << "\n"
        << " build_matches_EventName " << ( build_matches_EventName ? "YES" : "NO " ) << "\n"
        << sbuild::Desc()
        << " FIX by changing " << kEventName << " or rebuilding with suitable config "
        ;

    assert(build_matches_EventName);
    if(!build_matches_EventName) std::raise(SIGINT);
}



/**
SEventConfig::Initialize_Comp
-----------------------------

Invoked by SEventConfig::Initialize
AND by SetRGMode when the RG mode is changed.
That is not a normal thing to do, but can happen
when doing EndOfRun simtracing.

**/


void SEventConfig::Initialize_Comp()
{
    unsigned gather_mask = 0 ;
    unsigned save_mask = 0 ;

    if(     IsRGModeSimulate()) Initialize_Comp_Simulate_(gather_mask, save_mask);
    else if(IsRGModeSimtrace()) Initialize_Comp_Simtrace_(gather_mask, save_mask);
    else if(IsRGModeRender())   Initialize_Comp_Render_(gather_mask, save_mask);

    SetGatherComp_(gather_mask);
    SetSaveComp_(  save_mask);
}


/**
SEventConfig::Initialize_Comp_Simulate_
----------------------------------------

Canonically invoked by SEventConfig::Initialize_Comp

enum values like SCOMP_PHOTON are bitwise-ORed into the
gather and save masks based on configured MAX values.

**/

void SEventConfig::Initialize_Comp_Simulate_(unsigned& gather_mask, unsigned& save_mask )
{
    const char* mode = EventMode();
    int record_limit = RecordLimit();
    LOG(LEVEL)
        << " EventMode() " << mode     // eg Default, DebugHeavy
        << " RunningMode() " << RunningMode()
        << " RunningModeLabel() " << RunningModeLabel()
        << " record_limit " << record_limit
        ;

    if(IsDebugHeavy())
    {
        SEventConfig::SetMaxRec(0);
        SEventConfig::SetMaxRecord(record_limit);
        SEventConfig::SetMaxPrd(record_limit);
        SEventConfig::SetMaxAux(record_limit);

        SEventConfig::SetMaxSeq(1);
        // since moved to compound sflat/stag so MaxFlat/MaxTag should now either be 0 or 1, nothing else
        SEventConfig::SetMaxTag(1);
        SEventConfig::SetMaxFlat(1);
        SEventConfig::SetMaxSup(1);

    }
    else if(IsDebugLite())
    {
        SEventConfig::SetMaxRec(0);
        SEventConfig::SetMaxRecord(record_limit);
        SEventConfig::SetMaxSeq(1);  // formerly incorrectly set to max_bounce+1
    }




    if(IsNothing())
    {
        LOG(LEVEL) << "IsNothing()" ;
        gather_mask = 0 ;
        save_mask = 0 ;
    }
    else if(IsMinimal())
    {
        LOG(LEVEL) << "IsMinimal()" ;
        gather_mask = SCOMP_HIT ;
        save_mask = 0 ;
    }
    else if(IsHit())
    {
        LOG(LEVEL) << "IsHit()" ;
        gather_mask = SCOMP_HIT | SCOMP_GENSTEP ;
        save_mask = SCOMP_HIT | SCOMP_GENSTEP ;
    }
    else if(IsHitPhoton())
    {
        LOG(LEVEL) << "IsHitPhoton()" ;
        gather_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_GENSTEP  ;
        save_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_GENSTEP ;
    }
    else if(IsHitPhotonSeq())
    {
        LOG(LEVEL) << "IsHitPhotonSeq()" ;
        gather_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_SEQ | SCOMP_GENSTEP  ;
        save_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_SEQ | SCOMP_GENSTEP ;
        SetMaxSeq(1);
    }
    else if(IsHitSeq())
    {
        LOG(LEVEL) << "IsHitSeq()" ;
        gather_mask = SCOMP_HIT | SCOMP_SEQ | SCOMP_GENSTEP  ;
        save_mask = SCOMP_HIT | SCOMP_SEQ | SCOMP_GENSTEP ;
        SetMaxSeq(1);
    }
    else
    {
        gather_mask |= SCOMP_DOMAIN ;  save_mask |= SCOMP_DOMAIN ;

        if(MaxGenstep()>0){  gather_mask |= SCOMP_GENSTEP ; save_mask |= SCOMP_GENSTEP ; }
        if(MaxPhoton()>0)
        {
            gather_mask |= SCOMP_INPHOTON ;  save_mask |= SCOMP_INPHOTON ;
            gather_mask |= SCOMP_PHOTON   ;  save_mask |= SCOMP_PHOTON   ;
            gather_mask |= SCOMP_HIT      ;  save_mask |= SCOMP_HIT ;
            //gather_mask |= SCOMP_SEED ;   save_mask |= SCOMP_SEED ;  // only needed for deep debugging
        }
        if(MaxRecord()>0){    gather_mask |= SCOMP_RECORD ;  save_mask |= SCOMP_RECORD ; }
        if(MaxAux()>0){       gather_mask |= SCOMP_AUX    ;  save_mask |= SCOMP_AUX    ; }
        if(MaxSup()>0){       gather_mask |= SCOMP_SUP    ;  save_mask |= SCOMP_SUP    ; }
        if(MaxSeq()>0){       gather_mask |= SCOMP_SEQ    ;  save_mask |= SCOMP_SEQ    ; }
        if(MaxPrd()>0){       gather_mask |= SCOMP_PRD    ;  save_mask |= SCOMP_PRD    ; }
        if(MaxTag()>0){       gather_mask |= SCOMP_TAG    ;  save_mask |= SCOMP_TAG    ; }
        if(MaxFlat()>0){      gather_mask |= SCOMP_FLAT   ;  save_mask |= SCOMP_FLAT   ; }
    }

    if(IsRunningModeG4StateSave() || IsRunningModeG4StateRerun())
    {
        LOG(LEVEL) << " adding SCOMP_G4STATE to comp list " ;
        gather_mask |= SCOMP_G4STATE ; save_mask |= SCOMP_G4STATE ;
    }
    else
    {
        LOG(LEVEL) << " NOT : adding SCOMP_G4STATE to comp list " ;
    }
}

void SEventConfig::Initialize_Comp_Simtrace_(unsigned& gather_mask, unsigned& save_mask )
{
    assert(IsRGModeSimtrace());
    if(MaxGenstep()>0){   gather_mask |= SCOMP_GENSTEP  ;  save_mask |= SCOMP_GENSTEP ;  }
    if(MaxSimtrace()>0){  gather_mask |= SCOMP_SIMTRACE ;  save_mask |= SCOMP_SIMTRACE ; }

    LOG(info)
        << " MaxGenstep " << MaxGenstep()
        << " MaxSimtrace " << MaxSimtrace()
        << " gather_mask " << gather_mask
        << " save_mask " << save_mask
        ;

}
void SEventConfig::Initialize_Comp_Render_(unsigned& gather_mask, unsigned& save_mask )
{
    assert(IsRGModeRender());
    gather_mask |= SCOMP_PIXEL ;  save_mask |=  SCOMP_PIXEL ;
}

/**
SEventConfig::Serialize
-----------------------

Called for example from SEvt::addEventConfigArray

**/


NP* SEventConfig::Serialize() // static
{
    NP* meta = NP::Make<int>(1) ;

    const char* em = EventMode();
    if(em)  meta->set_meta<std::string>("EventMode", em );

    meta->set_meta<int>("RunningMode", RunningMode() );

    const char* rml = RunningModeLabel();
    if(rml) meta->set_meta<std::string>("RunningModeLabel", rml );

    const char* g4s = G4StateSpec() ;
    if(g4s) meta->set_meta<std::string>("G4StateSpec", g4s );

    meta->set_meta<int>("G4StateRerun", G4StateRerun() );
    meta->set_meta<int>("MaxCurand", MaxCurand() );
    meta->set_meta<int>("MaxSlot", MaxSlot() );
    meta->set_meta<int>("MaxGenstep", MaxGenstep() );
    meta->set_meta<int>("MaxPhoton", MaxPhoton() );
    meta->set_meta<int>("MaxSimtrace", MaxSimtrace() );

    meta->set_meta<int>("MaxBounce", MaxBounce() );
    meta->set_meta<float>("MaxTime", MaxTime() );

    meta->set_meta<int>("MaxRecord", MaxRecord() );
    meta->set_meta<int>("MaxRec", MaxRec() );
    meta->set_meta<int>("MaxAux", MaxAux() );
    meta->set_meta<int>("MaxSup", MaxSup() );
    meta->set_meta<int>("MaxSeq", MaxSeq() );
    meta->set_meta<int>("MaxPrd", MaxPrd() );
    meta->set_meta<int>("MaxTag", MaxTag() );
    meta->set_meta<int>("MaxFlat", MaxFlat() );

    meta->set_meta<float>("MaxExtentDomain", MaxExtentDomain() );
    meta->set_meta<float>("MaxTimeDomain", MaxTimeDomain() );

    const char* of = OutFold() ;
    if(of) meta->set_meta<std::string>("OutFold", of );

    const char* on = OutName() ;
    if(on) meta->set_meta<std::string>("OutName", on );

    meta->set_meta<unsigned>("HitMask", HitMask() );
    meta->set_meta<std::string>("HitMaskLabel", HitMaskLabel() );

    meta->set_meta<unsigned>("GatherComp", GatherComp() );
    meta->set_meta<unsigned>("SaveComp", SaveComp() );

    meta->set_meta<std::string>("DescGatherComp", DescGatherComp());
    meta->set_meta<std::string>("DescSaveComp", DescSaveComp());

    meta->set_meta<float>("PropagateEpsilon", PropagateEpsilon() );
    meta->set_meta<float>("PropagateEpsilon0", PropagateEpsilon0() );


    const char* ig  = InputGenstep() ;
    if(ig)  meta->set_meta<std::string>("InputGenstep", ig );

    const char* igsel  = InputGenstepSelection() ;
    if(igsel)  meta->set_meta<std::string>("InputGenstepSelection", igsel );


    const char* ip  = InputPhoton() ;
    if(ip)  meta->set_meta<std::string>("InputPhoton", ip );

    const char* ipf = InputPhotonFrame() ;
    if(ipf) meta->set_meta<std::string>("InputPhotonFrame", ipf );

    meta->set_meta<float>("InputPhotonChangeTime", InputPhotonChangeTime() );


    meta->set_meta<int>("RGMode", RGMode() );

    const char* rgml = RGModeLabel() ;
    if(rgml) meta->set_meta<std::string>("RGModeLabel", rgml );


    return meta ;
}

void SEventConfig::Save(const char* dir ) // static
{
    if(dir == nullptr) return ;
    NP* meta =Serialize();
    meta->save(dir, NAME );
}

/**
SEventConfig::SetDevice
-------------------------

Invoked at first SEvt instanciation::

    SEventConfig::SetDevice
    scontext::initConfig
    scontext::init
    scontext::scontext
    SEventConfig::Initialize_Meta
    SEventConfig::Initialize
    SEvt::SEvt
    SEvt::Create

Maximum number of photon slots that can be simulated in a single GPU
launch depends on:

1. VRAM, available from scontext.h invoked by CSGOptiX::Create

2. sizeof(curandState)+photon 4x4x4 + plus optional enabled arrays
   configured in SEventConfig::

   sizeof(curandStateXORWOW) 48
   sizeof(sphoton)           64=4*4*4
   .                       -----
   .                        112 bytes per photon (absolute minimum)

3. limited by available (chunked) curandState, see QRng.hh SCurandState.h
   (currently M200). BUT other than consuming disk it is perfectly possible
   to curand_init more chunks of curandState than could ever be used in
   currently available GPU VRAM (48GB, 80GB)
   because many launches are done due to curand_init taking lots of stack.

   * for now could just assert in QRng that maxphoton is less than
     the available curandState slots

4. safety scaledown from theoretical maximum for reliability,
   as detector geometry will take a few GB plus other processes
   will use some too

   * HMM: could access current free VRAM also, but that could change
     between the check and the launch


Experience with 48G GPU (48*1024*1024*1024 = 51539607552)::

    400M photons with 48*1024*1024*1024

    In [3]: 48*1024*1024*1024/112/1e6
    Out[3]: 460.1750674285714

    In [6]: 48*1024*1024*1024/112/1e6*0.9
    Out[6]: 414.1575606857143

    In [5]: 48*1024*1024*1024/112/1e6*0.87
    Out[5]: 400.3523086628571


Assuming get to use 90% of total VRAM in ballpark of observed 400M limit

TODO: measure total VRAM usage during large photon number scan to
provide some parameters to use in a better heuristic and get idea
of variability. Expect linear with some pedestal.

**/

void SEventConfig::SetDevice( size_t totalGlobalMem_bytes, std::string name )
{
    SetDeviceName( name.empty() ? nullptr : name.c_str() ) ;
    LOG(info) << DescDevice(totalGlobalMem_bytes, name) ;

    size_t mxs0  = MaxSlot();
    size_t hmxr = HeuristicMaxSlot_Rounded(totalGlobalMem_bytes);

    bool MaxSlot_is_zero = mxs0 == 0 ;
    if(MaxSlot_is_zero) SetMaxSlot(hmxr)
;
    size_t mxs1  = MaxSlot();

    bool changed = mxs1 != mxs0  ;

    LOG(info)
        << " Configured_MaxSlot/M " << mxs0/M
        << " Final_MaxSlot/M " << mxs1/M
        << " HeuristicMaxSlot_Rounded/M "  << hmxr/M
        << " changed " << ( changed ? "YES" : "NO " )
        << " DeviceName " << ( DeviceName() ? DeviceName() : "-" )
        << " HasDevice " << ( HasDevice() ? "YES" : "NO " )
        << "\n"
        << "(export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) " ;
        ;
}


void SEventConfig::SetDeviceName( const char* name )
{
    _DeviceName = name ? strdup(name) : nullptr ;
}


/**
SEventConfig::HeuristicMaxSlot
-------------------------------

Currently no accounting for the configured debug arrays.
Are assuming production type running.

For example the record array will scale memory per photon
by factor of ~32 from the 32 step points
(not accounting for curandState).

When debug arrays are configured the user currently
needs to manually keep total photon count low (few millions)
to stay within VRAM.  See::

    QSimTest::fake_propagate
    QSimTest::EventConfig

**/
size_t SEventConfig::HeuristicMaxSlot( size_t totalGlobalMem_bytes )
{
    return size_t(float(totalGlobalMem_bytes)*0.87f/112.f) ;
}

/**
SEventConfig::HeuristicMaxSlot_Rounded
-----------------------------------------

Rounded down to nearest million.

**/

size_t SEventConfig::HeuristicMaxSlot_Rounded( size_t totalGlobalMem_bytes )
{
    size_t hmx = HeuristicMaxSlot(totalGlobalMem_bytes);
    size_t hmx_M = hmx/M ;
    return hmx_M*M ;
}

std::string SEventConfig::DescDevice(size_t totalGlobalMem_bytes, std::string name )  // static
{
    size_t hmx = HeuristicMaxSlot(totalGlobalMem_bytes);
    size_t hmx_M = hmx/M ;
    size_t hmxr = HeuristicMaxSlot_Rounded(totalGlobalMem_bytes);
    size_t mxs  = MaxSlot();

    int wid = 35 ;
    std::stringstream ss ;
    ss << "SEventConfig::DescDevice"
       << "\n"
       << std::setw(wid) << "name                             : " << name
       << "\n"
       << std::setw(wid) << "totalGlobalMem_bytes             : " << totalGlobalMem_bytes
       << "\n"
       << std::setw(wid) << "totalGlobalMem_GB                : " << totalGlobalMem_bytes/(1024*1024*1024)
       << "\n"
       << std::setw(wid) << "HeuristicMaxSlot(VRAM)           : " << hmx
       << "\n"
       << std::setw(wid) << "HeuristicMaxSlot(VRAM)/M         : " << hmx_M
       << "\n"
       << std::setw(wid) << "HeuristicMaxSlot_Rounded(VRAM)   : " << hmxr
       << "\n"
       << std::setw(wid) << "MaxSlot/M                        : " << mxs/M
       << "\n"
       ;

    std::string str = ss.str() ;
    return str ;
}


salloc* SEventConfig::AllocEstimate(int _max_slot)
{
    uint64_t max_slot = _max_slot == 0 ? MaxSlot() : _max_slot ;
    uint64_t max_record = MaxRecord();
    uint64_t max_genstep = MaxGenstep();

    salloc* estimate = new salloc ;
    estimate->add("QEvent::setGenstep/device_alloc_genstep_and_seed:quad6/max_genstep", max_genstep, sizeof(quad6) ) ;
    estimate->add("QEvent::setGenstep/device_alloc_genstep_and_seed:int/max_slot", max_slot, sizeof(int));
    estimate->add("QEvent::device_alloc_photon/max_slot*sizeof(sphoton)", max_slot, sizeof(sphoton)) ;
    if(GatherRecord()) estimate->add("QEvent::device_alloc_photon/max_slot*max_record*sizeof(sphoton)", max_slot*max_record, sizeof(sphoton) );

    return estimate ;
}

uint64_t SEventConfig::AllocEstimateTotal(int _max_slot)
{
    salloc* estimate = AllocEstimate(_max_slot);
    uint64_t total = estimate->get_total();
    delete estimate ;
    return total ;
}

