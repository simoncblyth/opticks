#include <sstream>
#include <cstring>
#include <csignal>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ssys.h"
#include "sstr.h"
#include "spath.h"
#include "sdirectory.h"
#include "salloc.h"

#include "SPath.hh"   // only SPath::Make to replace

#include "SEventConfig.hh"
#include "SRG.h"  // raygenmode
#include "SRM.h"  // runningmode
#include "SComp.h"
#include "OpticksPhoton.hh"

#include "SLOG.hh"

const plog::Severity SEventConfig::LEVEL = SLOG::EnvLevel("SEventConfig", "DEBUG") ; 

int         SEventConfig::_IntegrationModeDefault = -1 ;
const char* SEventConfig::_EventModeDefault = "Default" ; 
const char* SEventConfig::_RunningModeDefault = "SRM_DEFAULT" ;
int         SEventConfig::_StartIndexDefault = 0 ;
int         SEventConfig::_NumEventDefault = 1 ;
const char* SEventConfig::_NumPhotonDefault = nullptr ;
const char* SEventConfig::_G4StateSpecDefault = "1000:38" ;
const char* SEventConfig::_G4StateSpecNotes   = "38=2*17+4 is appropriate for MixMaxRng" ; 
int         SEventConfig::_G4StateRerunDefault = -1 ;
const char* SEventConfig::_MaxBounceNotes = "MaxBounceLimit:31, MaxRecordLimit:32 (see sseq.h)" ; 
 
int SEventConfig::_MaxBounceDefault = 9 ; 

int SEventConfig::_MaxRecordDefault = 0 ; 
int SEventConfig::_MaxRecDefault = 0 ; 
int SEventConfig::_MaxAuxDefault = 0 ; 
int SEventConfig::_MaxSupDefault = 0 ; 
int SEventConfig::_MaxSeqDefault = 0 ; 
int SEventConfig::_MaxPrdDefault = 0 ; 
int SEventConfig::_MaxTagDefault = 0 ; 
int SEventConfig::_MaxFlatDefault = 0 ; 
float SEventConfig::_MaxExtentDefault = 1000.f ;  // mm  : domain compression used by *rec*
float SEventConfig::_MaxTimeDefault = 10.f ; // ns 
const char* SEventConfig::_OutFoldDefault = "$DefaultOutputDir" ; 
const char* SEventConfig::_OutNameDefault = nullptr ; 
const char* SEventConfig::_RGModeDefault = "simulate" ; 
const char* SEventConfig::_HitMaskDefault = "SD" ; 

#ifdef __APPLE__
const char* SEventConfig::_MaxGenstepDefault = "M1" ; 
const char* SEventConfig::_MaxPhotonDefault = "M1" ; 
const char* SEventConfig::_MaxSimtraceDefault = "M1" ; 
#else
const char* SEventConfig::_MaxGenstepDefault = "M3" ; 
const char* SEventConfig::_MaxPhotonDefault = "M3" ; 
const char* SEventConfig::_MaxSimtraceDefault = "M3" ; 
#endif


const char* SEventConfig::_GatherCompDefault = SComp::ALL_ ; 
const char* SEventConfig::_SaveCompDefault = SComp::ALL_ ; 

float SEventConfig::_PropagateEpsilonDefault = 0.05f ; 

const char* SEventConfig::_InputGenstepDefault = nullptr ; 
const char* SEventConfig::_InputPhotonDefault = nullptr ; 
const char* SEventConfig::_InputPhotonFrameDefault = nullptr ; 


int         SEventConfig::_IntegrationMode = ssys::getenvint(kIntegrationMode, _IntegrationModeDefault ); 
const char* SEventConfig::_EventMode = ssys::getenvvar(kEventMode, _EventModeDefault ); 
int         SEventConfig::_RunningMode = SRM::Type(ssys::getenvvar(kRunningMode, _RunningModeDefault)); 
int         SEventConfig::_StartIndex = ssys::getenvint(kStartIndex, _StartIndexDefault ); 
int         SEventConfig::_NumEvent = ssys::getenvint(kNumEvent, _NumEventDefault ); 

std::vector<int>* SEventConfig::_GetNumPhotonPerEvent()
{
    const char* NumPhotonSpec = ssys::getenvvar(kNumPhoton,  _NumPhotonDefault ); 
    return sstr::ParseIntSpecList<int>( NumPhotonSpec, ',' ); 
}
std::vector<int>* SEventConfig::_NumPhotonPerEvent = _GetNumPhotonPerEvent() ; 

/**

OPTICKS_NUM_PHOTON=M1,2,3,4 OPTICKS_NUM_EVENT=4 SEventConfigTest 

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

/**

::

    OPTICKS_NUM_EVENT=4 OPTICKS_NUM_PHOTON=M1:3 SEventConfigTest 

**/

int SEventConfig::_GetNumEvent()
{
    bool have_NumPhotonPerEvent = _NumPhotonPerEvent && _NumPhotonPerEvent->size() > 0 ; 
    bool override_NumEvent = have_NumPhotonPerEvent && int(_NumPhotonPerEvent->size()) != _NumEvent ; 
    LOG_IF(error, override_NumEvent ) 
        << " Overriding NumEvent "
        << "(" << kNumEvent  << ")" 
        << " value " << _NumEvent 
        << " as inconsistent with NumPhoton list length "
        << "(" << kNumPhoton << ")" 
        << " of " << ( _NumPhotonPerEvent ? _NumPhotonPerEvent->size() : -1 )
        ;
    return override_NumEvent ? int(_NumPhotonPerEvent->size()) : _NumEvent ; 
}


int SEventConfig::NumPhoton(int idx){ return _GetNumPhoton(idx) ; }
int SEventConfig::NumEvent(){         return _GetNumEvent() ; }
int SEventConfig::EventIndex(int idx){ return _StartIndex + idx ; }
int SEventConfig::EventIndexArg(int index){ return index == MISSING_INDEX ? -1 : index - _StartIndex ; }

bool SEventConfig::IsFirstEvent(int idx){ return idx == 0 ; }  // 0-based idx (such as Geant4 eventID)
bool SEventConfig::IsLastEvent(int idx){ return idx == NumEvent()-1 ; }  // 0-based idx (such as Geant4 eventID)


const char* SEventConfig::_G4StateSpec  = ssys::getenvvar(kG4StateSpec,  _G4StateSpecDefault ); 
int         SEventConfig::_G4StateRerun = ssys::getenvint(kG4StateRerun, _G4StateRerunDefault) ; 

int SEventConfig::_MaxGenstep   = ssys::getenv_ParseInt(kMaxGenstep,  _MaxGenstepDefault ) ; 
int SEventConfig::_MaxPhoton    = ssys::getenv_ParseInt(kMaxPhoton,   _MaxPhotonDefault ) ; 
int SEventConfig::_MaxSimtrace  = ssys::getenv_ParseInt(kMaxSimtrace, _MaxSimtraceDefault ) ; 

int SEventConfig::_MaxBounce    = ssys::getenvint(kMaxBounce, _MaxBounceDefault ) ; 
int SEventConfig::_MaxRecord    = ssys::getenvint(kMaxRecord, _MaxRecordDefault ) ;    
int SEventConfig::_MaxRec       = ssys::getenvint(kMaxRec, _MaxRecDefault ) ;   
int SEventConfig::_MaxAux       = ssys::getenvint(kMaxAux, _MaxAuxDefault ) ;   
int SEventConfig::_MaxSup       = ssys::getenvint(kMaxSup, _MaxSupDefault ) ;   
int SEventConfig::_MaxSeq       = ssys::getenvint(kMaxSeq,  _MaxSeqDefault ) ;  
int SEventConfig::_MaxPrd       = ssys::getenvint(kMaxPrd,  _MaxPrdDefault ) ;  
int SEventConfig::_MaxTag       = ssys::getenvint(kMaxTag,  _MaxTagDefault ) ;  
int SEventConfig::_MaxFlat      = ssys::getenvint(kMaxFlat,  _MaxFlatDefault ) ;  
float SEventConfig::_MaxExtent  = ssys::getenvfloat(kMaxExtent, _MaxExtentDefault );  
float SEventConfig::_MaxTime    = ssys::getenvfloat(kMaxTime,   _MaxTimeDefault );    // ns
const char* SEventConfig::_OutFold = ssys::getenvvar(kOutFold, _OutFoldDefault ); 
const char* SEventConfig::_OutName = ssys::getenvvar(kOutName, _OutNameDefault ); 
int SEventConfig::_RGMode = SRG::Type(ssys::getenvvar(kRGMode, _RGModeDefault)) ;    
unsigned SEventConfig::_HitMask  = OpticksPhoton::GetHitMask(ssys::getenvvar(kHitMask, _HitMaskDefault )) ;   

unsigned SEventConfig::_GatherComp  = SComp::Mask(ssys::getenvvar(kGatherComp, _GatherCompDefault )) ;   
unsigned SEventConfig::_SaveComp    = SComp::Mask(ssys::getenvvar(kSaveComp,   _SaveCompDefault )) ;   


float SEventConfig::_PropagateEpsilon = ssys::getenvfloat(kPropagateEpsilon, _PropagateEpsilonDefault ) ; 

const char* SEventConfig::_InputGenstep = ssys::getenvvar(kInputGenstep, _InputGenstepDefault ); 
const char* SEventConfig::_InputPhoton = ssys::getenvvar(kInputPhoton, _InputPhotonDefault ); 
const char* SEventConfig::_InputPhotonFrame = ssys::getenvvar(kInputPhotonFrame, _InputPhotonFrameDefault ); 


int         SEventConfig::IntegrationMode(){ return _IntegrationMode ; }
bool        SEventConfig::GPU_Simulation(){  return _IntegrationMode == 1 || _IntegrationMode == 3 ; }
bool        SEventConfig::CPU_Simulation(){  return _IntegrationMode == 2 || _IntegrationMode == 3 ; }

const char* SEventConfig::EventMode(){ return _EventMode ; }


int         SEventConfig::RunningMode(){ return _RunningMode ; }
const char* SEventConfig::RunningModeLabel(){ return SRM::Name(_RunningMode) ; }

bool SEventConfig::IsRunningModeDefault(){      return RunningMode() == SRM_DEFAULT ; } 
bool SEventConfig::IsRunningModeG4StateSave(){  return RunningMode() == SRM_G4STATE_SAVE ; } 
bool SEventConfig::IsRunningModeG4StateRerun(){ return RunningMode() == SRM_G4STATE_RERUN ; } 

bool SEventConfig::IsRunningModeTorch(){         return RunningMode() == SRM_TORCH ; } 
bool SEventConfig::IsRunningModeInputPhoton(){   return RunningMode() == SRM_INPUT_PHOTON ; } 
bool SEventConfig::IsRunningModeInputGenstep(){  return RunningMode() == SRM_INPUT_GENSTEP ; } 
bool SEventConfig::IsRunningModeGun(){           return RunningMode() == SRM_GUN ; } 
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


int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
int SEventConfig::MaxSimtrace(){   return _MaxSimtrace ; }
int SEventConfig::MaxCurandState(){ return std::max( MaxPhoton(), MaxSimtrace() ) ; }

int SEventConfig::MaxBounce(){   return _MaxBounce ; }
int SEventConfig::MaxRecord(){   return _MaxRecord ; }
int SEventConfig::MaxRec(){      return _MaxRec ; }
int SEventConfig::MaxAux(){      return _MaxAux ; }
int SEventConfig::MaxSup(){      return _MaxSup ; }
int SEventConfig::MaxSeq(){      return _MaxSeq ; }
int SEventConfig::MaxPrd(){      return _MaxPrd ; }
int SEventConfig::MaxTag(){      return _MaxTag ; }
int SEventConfig::MaxFlat(){      return _MaxFlat ; }
float SEventConfig::MaxExtent(){ return _MaxExtent ; }
float SEventConfig::MaxTime(){   return _MaxTime ; }
const char* SEventConfig::OutFold(){   return _OutFold ; }
const char* SEventConfig::OutName(){   return _OutName ; }
unsigned SEventConfig::HitMask(){     return _HitMask ; }

unsigned SEventConfig::GatherComp(){  return _GatherComp ; } 
unsigned SEventConfig::SaveComp(){    return _SaveComp ; } 


float SEventConfig::PropagateEpsilon(){ return _PropagateEpsilon ; }


const char* SEventConfig::_InputGenstepPath(int idx)
{
    std::string str = sstr::Format_(_InputGenstep, idx ) ; 
    return strdup(str.c_str()) ; 
}
const char* SEventConfig::InputGenstep(int idx)
{
    return ( idx == -1 || _InputGenstep == nullptr ) ? _InputGenstep : _InputGenstepPath(idx) ; 
}

bool SEventConfig::InputGenstepPathExists(int idx)
{
    const char* path = SEventConfig::InputGenstep(idx); 
    return path ? spath::Exists(path) : false ;  
}


const char* SEventConfig::InputPhoton(){   return _InputPhoton ; }
const char* SEventConfig::InputPhotonFrame(){   return _InputPhotonFrame ; }


int SEventConfig::RGMode(){  return _RGMode ; } 
bool SEventConfig::IsRGModeRender(){   return RGMode() == SRG_RENDER   ; } 
bool SEventConfig::IsRGModeSimtrace(){ return RGMode() == SRG_SIMTRACE ; } 
bool SEventConfig::IsRGModeSimulate(){ return RGMode() == SRG_SIMULATE ; } 
bool SEventConfig::IsRGModeTest(){     return RGMode() == SRG_TEST ; } 
const char* SEventConfig::RGModeLabel(){ return SRG::Name(_RGMode) ; }



 
void SEventConfig::SetDefault(){            SetEventMode(Default)           ; } 
void SEventConfig::SetDebugHeavy(){         SetEventMode(DebugHeavy) ; }
void SEventConfig::SetDebugLite(){          SetEventMode(DebugLite) ; }
void SEventConfig::SetNothing(){            SetEventMode(Nothing)           ; }
void SEventConfig::SetMinimal(){            SetEventMode(Minimal)           ; }
void SEventConfig::SetHit(){                SetEventMode(Hit)               ; }
void SEventConfig::SetHitPhoton(){          SetEventMode(HitPhoton)         ; }
void SEventConfig::SetHitPhotonSeq(){       SetEventMode(HitPhotonSeq)      ; }
void SEventConfig::SetHitSeq(){             SetEventMode(HitSeq)            ; }

bool SEventConfig::IsDefault(){           return _EventMode && strcmp(_EventMode, Default) == 0 ; }
bool SEventConfig::IsDebugHeavy(){        return _EventMode && strcmp(_EventMode, DebugHeavy) == 0 ; }
bool SEventConfig::IsDebugLite(){         return _EventMode && strcmp(_EventMode, DebugLite) == 0 ; }
bool SEventConfig::IsNothing(){           return _EventMode && strcmp(_EventMode, Nothing) == 0 ; }
bool SEventConfig::IsMinimal(){           return _EventMode && strcmp(_EventMode, Minimal) == 0 ; }
bool SEventConfig::IsHit(){               return _EventMode && strcmp(_EventMode, Hit) == 0 ; }
bool SEventConfig::IsHitPhoton(){         return _EventMode && strcmp(_EventMode, HitPhoton) == 0 ; }
bool SEventConfig::IsHitPhotonSeq(){      return _EventMode && strcmp(_EventMode, HitPhotonSeq) == 0 ; }
bool SEventConfig::IsHitSeq(){            return _EventMode && strcmp(_EventMode, HitSeq) == 0 ; }

std::string SEventConfig::DescEventMode()  // static
{
    std::stringstream ss ; 
    ss << "SEventConfig::DescEventMode" << std::endl 
       << Default 
       << std::endl
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


void SEventConfig::SetIntegrationMode(int mode){ _IntegrationMode = mode ; Check() ; }
void SEventConfig::SetEventMode(const char* mode){ _EventMode = mode ? strdup(mode) : nullptr ; Check() ; }
void SEventConfig::SetRunningMode(const char* mode){ _RunningMode = SRM::Type(mode) ; Check() ; }

void SEventConfig::SetStartIndex(int index0){        _StartIndex = index0 ; Check() ; }
void SEventConfig::SetNumEvent(int nevt){            _NumEvent = nevt ; Check() ; }
void SEventConfig::SetG4StateSpec(const char* spec){ _G4StateSpec = spec ? strdup(spec) : nullptr ; Check() ; }
void SEventConfig::SetG4StateRerun(int id){          _G4StateRerun = id ; Check() ; }

void SEventConfig::SetMaxGenstep(int max_genstep){ _MaxGenstep = max_genstep ; Check() ; }
void SEventConfig::SetMaxPhoton( int max_photon){  _MaxPhoton  = max_photon  ; Check() ; }
void SEventConfig::SetMaxSimtrace( int max_simtrace){  _MaxSimtrace  = max_simtrace  ; Check() ; }
void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
void SEventConfig::SetMaxRecord( int max_record){  _MaxRecord  = max_record  ; Check() ; }
void SEventConfig::SetMaxRec(    int max_rec){     _MaxRec     = max_rec     ; Check() ; }
void SEventConfig::SetMaxAux(    int max_aux){     _MaxAux     = max_aux     ; Check() ; }
void SEventConfig::SetMaxSup(    int max_sup){     _MaxSup     = max_sup     ; Check() ; }
void SEventConfig::SetMaxSeq(    int max_seq){     _MaxSeq     = max_seq     ; Check() ; }
void SEventConfig::SetMaxPrd(    int max_prd){     _MaxPrd     = max_prd     ; Check() ; }
void SEventConfig::SetMaxTag(    int max_tag){     _MaxTag     = max_tag     ; Check() ; }
void SEventConfig::SetMaxFlat(    int max_flat){     _MaxFlat     = max_flat     ; Check() ; }
void SEventConfig::SetMaxExtent( float max_extent){ _MaxExtent = max_extent  ; Check() ; }
void SEventConfig::SetMaxTime(   float max_time){   _MaxTime = max_time  ; Check() ; }
void SEventConfig::SetOutFold(   const char* outfold){   _OutFold = outfold ? strdup(outfold) : nullptr ; Check() ; }
void SEventConfig::SetOutName(   const char* outname){   _OutName = outname ? strdup(outname) : nullptr ; Check() ; }
void SEventConfig::SetHitMask(   const char* abrseq, char delim){  _HitMask = OpticksPhoton::GetHitMask(abrseq,delim) ; }

void SEventConfig::SetRGMode(   const char* mode){   _RGMode = SRG::Type(mode) ; Check() ; }
void SEventConfig::SetRGModeSimulate(){  SetRGMode( SRG::SIMULATE_ ); }
void SEventConfig::SetRGModeSimtrace(){  SetRGMode( SRG::SIMTRACE_ ); }
void SEventConfig::SetRGModeRender(){    SetRGMode( SRG::RENDER_ ); }
void SEventConfig::SetRGModeTest(){      SetRGMode( SRG::TEST_ ); }

void SEventConfig::SetPropagateEpsilon(float eps){ _PropagateEpsilon = eps ; Check() ; }

void SEventConfig::SetInputGenstep(const char* ig){   _InputGenstep = ig ? strdup(ig) : nullptr ; Check() ; }
void SEventConfig::SetInputPhoton(const char* ip){   _InputPhoton = ip ? strdup(ip) : nullptr ; Check() ; }
void SEventConfig::SetInputPhotonFrame(const char* ip){   _InputPhotonFrame = ip ? strdup(ip) : nullptr ; Check() ; }

void SEventConfig::SetGatherComp_(unsigned mask){ _GatherComp = mask ; }
void SEventConfig::SetGatherComp(const char* names, char delim){  SetGatherComp_( SComp::Mask(names,delim)) ; }

void SEventConfig::SetSaveComp_(unsigned mask){ _SaveComp = mask ; }
void SEventConfig::SetSaveComp(const char* names, char delim){  SetSaveComp_( SComp::Mask(names,delim)) ; }


void SEventConfig::SetComp()
{
     unsigned gather_mask = 0 ; 
     unsigned save_mask = 0 ; 
     CompAuto(gather_mask, save_mask ); 
 
     SetGatherComp_(gather_mask); 
     SetSaveComp_(  save_mask); 
}


/**
SEventConfig::CompAuto
---------------------------

Canonically invoked by SEventConfig::Initialize

enum values like SCOMP_PHOTON are bitwise-ORed into the 
gather and save masks based on configured MAX values. 

**/

void SEventConfig::CompAuto(unsigned& gather_mask, unsigned& save_mask )
{
    if(IsRGModeSimulate() && IsNothing())
    {
        LOG(LEVEL) << "IsRGModeSimulate() && IsNothing()" ; 
        gather_mask = 0 ; 
        save_mask = 0 ;  
    }
    else if(IsRGModeSimulate() && IsMinimal())
    {
        LOG(LEVEL) << "IsRGModeSimulate() && IsMinimal()" ; 
        gather_mask = SCOMP_HIT ; 
        save_mask = 0 ;  
    }
    else if(IsRGModeSimulate() && IsHit())
    {
        LOG(LEVEL) << "IsRGModeSimulate() && IsHit()" ; 
        gather_mask = SCOMP_HIT | SCOMP_GENSTEP ; 
        save_mask = SCOMP_HIT | SCOMP_GENSTEP ;   
    }
    else if(IsRGModeSimulate() && IsHitPhoton())
    {
        LOG(LEVEL) << "IsRGModeSimulate() && IsHitPhoton()" ; 
        gather_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_GENSTEP  ; 
        save_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_GENSTEP ;   
    }
    else if(IsRGModeSimulate() && IsHitPhotonSeq())
    {
        LOG(LEVEL) << "IsRGModeSimulate() && IsHitPhotonSeq()" ; 
        gather_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_SEQ | SCOMP_GENSTEP  ; 
        save_mask = SCOMP_HIT | SCOMP_PHOTON | SCOMP_SEQ | SCOMP_GENSTEP ;   
        SetMaxSeq(1);  
    }
    else if(IsRGModeSimulate() && IsHitSeq())
    {
        LOG(LEVEL) << "IsRGModeSimulate() && IsHitSeq()" ; 
        gather_mask = SCOMP_HIT | SCOMP_SEQ | SCOMP_GENSTEP  ; 
        save_mask = SCOMP_HIT | SCOMP_SEQ | SCOMP_GENSTEP ;   
        SetMaxSeq(1);  
    }
    else if(IsRGModeSimulate())
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
    else if(IsRGModeSimtrace())
    {
        if(MaxGenstep()>0){   gather_mask |= SCOMP_GENSTEP  ;  save_mask |= SCOMP_GENSTEP ;  }
        if(MaxSimtrace()>0){  gather_mask |= SCOMP_SIMTRACE ;  save_mask |= SCOMP_SIMTRACE ; }

        LOG(LEVEL) 
            << " MaxGenstep " << MaxGenstep()
            << " MaxSimtrace " << MaxSimtrace()
            << " gather_mask " << gather_mask 
            << " save_mask " << save_mask 
            ;

    } 
    else if(IsRGModeRender())
    {
        gather_mask |= SCOMP_PIXEL ;  save_mask |=  SCOMP_PIXEL ;
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






std::string SEventConfig::HitMaskLabel(){  return OpticksPhoton::FlagMask( _HitMask ) ; }


//std::string SEventConfig::CompMaskLabel(){ return SComp::Desc( _CompMask ) ; }
std::string SEventConfig::GatherCompLabel(){ return SComp::Desc( _GatherComp ) ; }
std::string SEventConfig::SaveCompLabel(){   return SComp::Desc( _SaveComp ) ; }


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
SEventConfig::Check
----------------------

Since moved to compound stag/sflat in stag.h 
MaxTag/MaxFlat must now either be 0 or 1, nothing else.
Had a bug previously with MaxTag/MaxFlat 24 that 
caused huge memory allocations in debug event modes. 

**/

//const int SEventConfig::LIMIT = 16 ; 
const int SEventConfig::LIMIT = 32 ; 

void SEventConfig::Check()
{
   assert( _IntegrationMode >= -1 && _IntegrationMode <= 3 ); 

   assert( _MaxBounce >= 0 && _MaxBounce <  LIMIT ) ;   // TRY 0 : FOR DEBUG 
   assert( _MaxRecord >= 0 && _MaxRecord <= LIMIT ) ; 
   assert( _MaxRec    >= 0 && _MaxRec    <= LIMIT ) ; 
   assert( _MaxPrd    >= 0 && _MaxPrd    <= LIMIT ) ; 

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
       << std::setw(25) << kG4StateSpec
       << std::setw(20) << " G4StateSpec " << " : " << G4StateSpec() 
       << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " G4StateSpecNotes " << " : " << _G4StateSpecNotes
       << std::endl 
       << std::setw(25) << kG4StateRerun
       << std::setw(20) << " G4StateRerun " << " : " << G4StateRerun() 
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
       << std::setw(25) << ""
       << std::setw(20) << " MaxCurandState " << " : " << MaxCurandState() 
       << std::setw(20) << " MaxCurandState/M " << " : " << MaxCurandState()/M
       << std::endl 
       << std::setw(25) << kMaxBounce
       << std::setw(20) << " MaxBounce " << " : " << MaxBounce() 
       << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " MaxBounceNotes " << " : " << _MaxBounceNotes
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
       << std::setw(25) << kMaxExtent
       << std::setw(20) << " MaxExtent " << " : " << MaxExtent() 
       << std::endl 
       << std::setw(25) << kMaxTime
       << std::setw(20) << " MaxTime " << " : " << MaxTime() 
       << std::endl 
       << std::setw(25) << kRGMode
       << std::setw(20) << " RGMode " << " : " << RGMode() 
       << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " RGModeLabel " << " : " << RGModeLabel() 
       << std::endl 
       /*
       << std::setw(25) << kCompMask
       << std::setw(20) << " CompMask " << " : " << CompMask() 
       << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " CompMaskLabel " << " : " << CompMaskLabel() 
       */
       << std::setw(25) << kGatherComp
       << std::setw(20) << " GatherComp " << " : " << GatherComp() 
       << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " GatherCompLabel " << " : " << GatherCompLabel() 
       << std::endl 
       << std::setw(25) << kSaveComp
       << std::setw(20) << " SaveComp " << " : " << SaveComp() 
       << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " SaveCompLabel " << " : " << SaveCompLabel() 
       << std::endl 
       << std::setw(25) << kOutFold
       << std::setw(20) << " OutFold " << " : " << OutFold() 
       << std::endl 
       << std::setw(25) << kOutName
       << std::setw(20) << " OutName " << " : " << ( OutName() ? OutName() : "-" )  
       << std::endl 
       << std::setw(25) << kPropagateEpsilon
       << std::setw(20) << " PropagateEpsilon " << " : " << std::fixed << std::setw(10) << std::setprecision(4) << PropagateEpsilon() 
       << std::endl 
       << std::setw(25) << kInputGenstep
       << std::setw(20) << " InputGenstep " << " : " << ( InputGenstep() ? InputGenstep() : "-" )  
       << std::endl 
       << std::setw(25) << kInputPhoton
       << std::setw(20) << " InputPhoton " << " : " << ( InputPhoton() ? InputPhoton() : "-" )  
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

/**
SEventConfig::Initialize
-------------------------

Canonically invoked from SEvt::SEvt 

* NB must make any static call adjustments before SEvt instanciation 
  for them to have any effect 

Former "StandardFullDebug" renamed "DebugHeavy" 
is far too heavy for most debugging/validation, 
"DebugLite" mode with photon, record, seq,  genstep 
covers most needs.  

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
    // assert( Initialize_COUNT == 0); 
    Initialize_COUNT += 1 ; 

    const char* mode = EventMode(); 
    LOG(LEVEL) <<  " EventMode() " << mode ;  // eg Default, DebugHeavy
    LOG(LEVEL) 
        <<  " RunningMode() " << RunningMode() 
        <<  " RunningModeLabel() " << RunningModeLabel() 
        ; 

    int max_bounce = MaxBounce(); 

    if(IsDefault())
    {
        SetComp() ;
    }
    else if(IsNothing() || IsMinimal() || IsHit() || IsHitPhoton() || IsHitPhotonSeq() || IsHitSeq() )
    {
        SetComp() ;  
    }
    else if(IsDebugHeavy())
    {
        SEventConfig::SetMaxRec(0); 
        SEventConfig::SetMaxRecord(max_bounce+1); 
        SEventConfig::SetMaxSeq(1);  // formerly incorrectly set to max_bounce+1

        SEventConfig::SetMaxPrd(max_bounce+1); 
        SEventConfig::SetMaxAux(max_bounce+1); 
        // since moved to compound sflat/stag so MaxFlat/MaxTag should now either be 0 or 1, nothing else  
        SEventConfig::SetMaxTag(1);   
        SEventConfig::SetMaxFlat(1); 
        SEventConfig::SetMaxSup(1); 

        SetComp() ;   // comp set based on Max values   
    }
    else if(IsDebugLite())
    {
        SEventConfig::SetMaxRec(0); 
        SEventConfig::SetMaxRecord(max_bounce+1); 
        SEventConfig::SetMaxSeq(1);  // formerly incorrectly set to max_bounce+1

        SetComp() ;   // comp set based on Max values   
    }
    else
    {
        LOG(fatal) << "mode [" << mode << "] IS NOT RECOGNIZED "  ;         
        LOG(fatal) << " options " << std::endl << DescEventMode() ; 

        std::cerr << "mode [" << mode << "] IS NOT RECOGNIZED " << std::endl   ;         
        std::cerr << " options " << std::endl << DescEventMode() << std::endl ; 
        
        std::raise(SIGINT);  
    }

    LOG(LEVEL) << Desc() ; 
    return 0 ; 
}


uint64_t SEventConfig::EstimateAlloc()
{
    salloc* estimate = new salloc ; 
    uint64_t tot = estimate->get_total() ; 
    delete estimate ; 
    return tot ; 
}


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
    meta->set_meta<int>("MaxGenstep", MaxGenstep() );  
    meta->set_meta<int>("MaxPhoton", MaxPhoton() );  
    meta->set_meta<int>("MaxSimtrace", MaxSimtrace() );  
    meta->set_meta<int>("MaxCurandState", MaxCurandState() );  

    meta->set_meta<int>("MaxBounce", MaxBounce() );  
    meta->set_meta<int>("MaxRecord", MaxRecord() );  
    meta->set_meta<int>("MaxRec", MaxRec() );  
    meta->set_meta<int>("MaxAux", MaxAux() );  
    meta->set_meta<int>("MaxSup", MaxSup() );  
    meta->set_meta<int>("MaxSeq", MaxSeq() );  
    meta->set_meta<int>("MaxPrd", MaxPrd() );  
    meta->set_meta<int>("MaxTag", MaxTag() );  
    meta->set_meta<int>("MaxFlat", MaxFlat() );  
    meta->set_meta<float>("MaxExtent", MaxExtent() );  
    meta->set_meta<float>("MaxTime", MaxTime() );  

    const char* of = OutFold() ; 
    if(of) meta->set_meta<std::string>("OutFold", of );  

    const char* on = OutName() ; 
    if(on) meta->set_meta<std::string>("OutName", on );  

    meta->set_meta<unsigned>("HitMask", HitMask() );  

    meta->set_meta<unsigned>("GatherComp", GatherComp() );  
    meta->set_meta<unsigned>("SaveComp", SaveComp() );  

    meta->set_meta<std::string>("GatherCompLabel", GatherCompLabel()); 
    meta->set_meta<std::string>("SaveCompLabel", SaveCompLabel()); 

    meta->set_meta<float>("PropagateEpsilon", PropagateEpsilon() );  


    const char* ig  = InputGenstep() ;  
    if(ig)  meta->set_meta<std::string>("InputGenstep", ig );  

    const char* ip  = InputPhoton() ;  
    if(ip)  meta->set_meta<std::string>("InputPhoton", ip );  

    const char* ipf = InputPhotonFrame() ;  
    if(ipf) meta->set_meta<std::string>("InputPhotonFrame", ipf );  

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

