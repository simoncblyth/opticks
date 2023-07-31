#include <sstream>
#include <cstring>
#include <csignal>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "SSys.hh"
#include "SPath.hh"
#include "SEventConfig.hh"
#include "SRG.h"  // raygenmode
#include "SRM.h"  // runningmode
#include "SComp.h"
#include "OpticksPhoton.hh"
#include "salloc.h"

#include "SLOG.hh"

const plog::Severity SEventConfig::LEVEL = SLOG::EnvLevel("SEventConfig", "DEBUG") ; 

int         SEventConfig::_IntegrationModeDefault = -1 ;
const char* SEventConfig::_EventModeDefault = "Default" ; 
const char* SEventConfig::_RunningModeDefault = "SRM_DEFAULT" ;
const char* SEventConfig::_G4StateSpecDefault = "1000:38" ;
const char* SEventConfig::_G4StateSpecNotes   = "38=2*17+4 is appropriate for MixMaxRng" ; 
int         SEventConfig::_G4StateRerunDefault = -1 ;
const char* SEventConfig::_MaxBounceNotes = "MaxBounceLimit:31, MaxRecordLimit:32 (see sseq.h)" ; 
 
int SEventConfig::_MaxGenstepDefault = 1000*K ; 
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
int  SEventConfig::_MaxPhotonDefault = 1*M ; 
int  SEventConfig::_MaxSimtraceDefault = 1*M ; 
#else
int  SEventConfig::_MaxPhotonDefault = 3*M ; 
int  SEventConfig::_MaxSimtraceDefault = 3*M ; 
#endif


//const char* SEventConfig::_CompMaskDefault = SComp::ALL_ ; 
const char* SEventConfig::_GatherCompDefault = SComp::ALL_ ; 
const char* SEventConfig::_SaveCompDefault = SComp::ALL_ ; 

float SEventConfig::_PropagateEpsilonDefault = 0.05f ; 
const char* SEventConfig::_InputPhotonDefault = nullptr ; 
const char* SEventConfig::_InputPhotonFrameDefault = nullptr ; 


int         SEventConfig::_IntegrationMode = SSys::getenvint(kIntegrationMode, _IntegrationModeDefault ); 
const char* SEventConfig::_EventMode = SSys::getenvvar(kEventMode, _EventModeDefault ); 
int SEventConfig::_RunningMode = SRM::Type(SSys::getenvvar(kRunningMode, _RunningModeDefault)); 
const char* SEventConfig::_G4StateSpec  = SSys::getenvvar(kG4StateSpec,  _G4StateSpecDefault ); 
int         SEventConfig::_G4StateRerun = SSys::getenvint(kG4StateRerun, _G4StateRerunDefault) ; 


int SEventConfig::_MaxGenstep   = SSys::getenvint(kMaxGenstep,  _MaxGenstepDefault ) ; 
int SEventConfig::_MaxPhoton    = SSys::getenvint(kMaxPhoton,   _MaxPhotonDefault ) ; 
int SEventConfig::_MaxSimtrace  = SSys::getenvint(kMaxSimtrace,   _MaxSimtraceDefault ) ; 
int SEventConfig::_MaxBounce    = SSys::getenvint(kMaxBounce, _MaxBounceDefault ) ; 
int SEventConfig::_MaxRecord    = SSys::getenvint(kMaxRecord, _MaxRecordDefault ) ;    
int SEventConfig::_MaxRec       = SSys::getenvint(kMaxRec, _MaxRecDefault ) ;   
int SEventConfig::_MaxAux       = SSys::getenvint(kMaxAux, _MaxAuxDefault ) ;   
int SEventConfig::_MaxSup       = SSys::getenvint(kMaxSup, _MaxSupDefault ) ;   
int SEventConfig::_MaxSeq       = SSys::getenvint(kMaxSeq,  _MaxSeqDefault ) ;  
int SEventConfig::_MaxPrd       = SSys::getenvint(kMaxPrd,  _MaxPrdDefault ) ;  
int SEventConfig::_MaxTag       = SSys::getenvint(kMaxTag,  _MaxTagDefault ) ;  
int SEventConfig::_MaxFlat      = SSys::getenvint(kMaxFlat,  _MaxFlatDefault ) ;  
float SEventConfig::_MaxExtent  = SSys::getenvfloat(kMaxExtent, _MaxExtentDefault );  
float SEventConfig::_MaxTime    = SSys::getenvfloat(kMaxTime,   _MaxTimeDefault );    // ns
const char* SEventConfig::_OutFold = SSys::getenvvar(kOutFold, _OutFoldDefault ); 
const char* SEventConfig::_OutName = SSys::getenvvar(kOutName, _OutNameDefault ); 
int SEventConfig::_RGMode = SRG::Type(SSys::getenvvar(kRGMode, _RGModeDefault)) ;    
unsigned SEventConfig::_HitMask  = OpticksPhoton::GetHitMask(SSys::getenvvar(kHitMask, _HitMaskDefault )) ;   

//unsigned SEventConfig::_CompMask  = SComp::Mask(SSys::getenvvar(kCompMask, _CompMaskDefault )) ;   
unsigned SEventConfig::_GatherComp  = SComp::Mask(SSys::getenvvar(kGatherComp, _GatherCompDefault )) ;   
unsigned SEventConfig::_SaveComp    = SComp::Mask(SSys::getenvvar(kSaveComp,   _SaveCompDefault )) ;   


float SEventConfig::_PropagateEpsilon = SSys::getenvfloat(kPropagateEpsilon, _PropagateEpsilonDefault ) ; 
const char* SEventConfig::_InputPhoton = SSys::getenvvar(kInputPhoton, _InputPhotonDefault ); 
const char* SEventConfig::_InputPhotonFrame = SSys::getenvvar(kInputPhotonFrame, _InputPhotonFrameDefault ); 


int         SEventConfig::IntegrationMode(){ return _IntegrationMode ; }
const char* SEventConfig::EventMode(){ return _EventMode ; }


int         SEventConfig::RunningMode(){ return _RunningMode ; }
const char* SEventConfig::RunningModeLabel(){ return SRM::Name(_RunningMode) ; }
bool SEventConfig::IsRunningModeDefault(){      return RunningMode() == SRM_DEFAULT ; } 
bool SEventConfig::IsRunningModeG4StateSave(){  return RunningMode() == SRM_G4STATE_SAVE ; } 
bool SEventConfig::IsRunningModeG4StateRerun(){ return RunningMode() == SRM_G4STATE_RERUN ; } 

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

//unsigned SEventConfig::CompMask(){  return _CompMask; } 
unsigned SEventConfig::GatherComp(){  return _GatherComp ; } 
unsigned SEventConfig::SaveComp(){    return _SaveComp ; } 


float SEventConfig::PropagateEpsilon(){ return _PropagateEpsilon ; }
const char* SEventConfig::InputPhoton(){   return _InputPhoton ; }
const char* SEventConfig::InputPhotonFrame(){   return _InputPhotonFrame ; }


int SEventConfig::RGMode(){  return _RGMode ; } 
bool SEventConfig::IsRGModeRender(){   return RGMode() == SRG_RENDER   ; } 
bool SEventConfig::IsRGModeSimtrace(){ return RGMode() == SRG_SIMTRACE ; } 
bool SEventConfig::IsRGModeSimulate(){ return RGMode() == SRG_SIMULATE ; } 
const char* SEventConfig::RGModeLabel(){ return SRG::Name(_RGMode) ; }





const char* SEventConfig::Default = "Default" ; 
const char* SEventConfig::StandardFullDebug = "StandardFullDebug" ; 
void SEventConfig::SetDefault(){            SetEventMode(Default)           ; } 
void SEventConfig::SetStandardFullDebug(){  SetEventMode(StandardFullDebug) ; }

void SEventConfig::SetIntegrationMode(int mode){ _IntegrationMode = mode ; Check() ; }
void SEventConfig::SetEventMode(const char* mode){ _EventMode = mode ? strdup(mode) : nullptr ; Check() ; }
void SEventConfig::SetRunningMode(const char* mode){ _RunningMode = SRM::Type(mode) ; Check() ; }
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

void SEventConfig::SetPropagateEpsilon(float eps){ _PropagateEpsilon = eps ; Check() ; }
void SEventConfig::SetInputPhoton(const char* ip){   _InputPhoton = ip ? strdup(ip) : nullptr ; Check() ; }
void SEventConfig::SetInputPhotonFrame(const char* ip){   _InputPhotonFrame = ip ? strdup(ip) : nullptr ; Check() ; }

/*
void SEventConfig::SetCompMask_(unsigned mask){ _CompMask = mask ; }
void SEventConfig::SetCompMask(const char* names, char delim){  SetCompMask_( SComp::Mask(names,delim)) ; }
void SEventConfig::SetCompMaskAuto(){ SetCompMask_( CompMaskAuto() ) ; }
*/

void SEventConfig::SetGatherComp_(unsigned mask){ _GatherComp = mask ; }
void SEventConfig::SetGatherComp(const char* names, char delim){  SetGatherComp_( SComp::Mask(names,delim)) ; }

void SEventConfig::SetSaveComp_(unsigned mask){ _SaveComp = mask ; }
void SEventConfig::SetSaveComp(const char* names, char delim){  SetSaveComp_( SComp::Mask(names,delim)) ; }


void SEventConfig::SetCompAuto()
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

**/

void SEventConfig::CompAuto(unsigned& gather_mask, unsigned& save_mask )
{
    if(IsRGModeSimulate())
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

void SEventConfig::SaveCompList( std::vector<unsigned>& save_comp )
{
    SComp::CompListMask(save_comp, SaveComp() ); 
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

   assert( _MaxBounce >  0 && _MaxBounce <  LIMIT ) ;  
   assert( _MaxRecord >= 0 && _MaxRecord <= LIMIT ) ; 
   assert( _MaxRec    >= 0 && _MaxRec    <= LIMIT ) ; 
   assert( _MaxSeq    >= 0 && _MaxSeq    <= LIMIT ) ; 
   assert( _MaxPrd    >= 0 && _MaxPrd    <= LIMIT ) ; 

   assert( _MaxTag    >= 0 && _MaxTag    <= 1 ) ; 
   assert( _MaxFlat   >= 0 && _MaxFlat   <= 1 ) ; 
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
       << std::endl 
       << std::setw(25) << kMaxPhoton 
       << std::setw(20) << " MaxPhoton " << " : " << MaxPhoton() 
       << std::endl 
       << std::setw(25) << kMaxSimtrace 
       << std::setw(20) << " MaxSimtrace " << " : " << MaxSimtrace() 
       << std::setw(20) << " MaxCurandState " << " : " << MaxCurandState() 
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

TODO: rejig, it makes more sense for SEventConfig to be used from SPath via SOpticksResource::Get tokens not vice versa 

**/

const char* SEventConfig::OutDir()  
{
    return SPath::Resolve( OutFold(), OutName(), DIRPATH ); 
}
const char* SEventConfig::OutPath( const char* stem, int index, const char* ext )
{
     return SPath::Make( OutFold(), OutName(), stem, index, ext, FILEPATH);  // HMM: an InPath would use NOOP to not create the dir
}

std::string SEventConfig::DescOutPath(  const char* stem, int index, const char* ext ) 
{
    const char* path = OutPath(stem, index, ext ) ; 
    std::stringstream ss ; 
    ss << "SEventConfig::DescOutPath" << std::endl 
       << " stem " << ( stem ? stem : "-" )
       << " index " << index
       << " ext " << ( ext ? ext : "-" )
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
    return SPath::Resolve( OutFold(), OutName(), reldir, DIRPATH ); 
}
const char* SEventConfig::OutPath( const char* reldir, const char* stem, int index, const char* ext )
{
     return SPath::Make( OutFold(), OutName(), reldir, stem, index, ext, FILEPATH ); 
}

/**
SEventConfig::Initialize
-------------------------

Canonically invoked from SEvt::SEvt 

* Formerly this was invoked from G4CXOpticks::init, but that is 
  too high level as SEvt is needed for purely G4 running such as the U4 tests 

* NB must make any static call adjustments before SEvt instanciation 
  for them to have any effect 


TODO: check if still conflation between the comps to gather and comp existance ?
      need to split those, eg photon comp is always needed but not always gathered

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
    LOG(LEVEL) <<  " EventMode() " << mode ; 
    LOG(LEVEL) 
        <<  " RunningMode() " << RunningMode() 
        <<  " RunningModeLabel() " << RunningModeLabel() 
        ; 

    //std::raise(SIGINT); 

    int maxbounce = MaxBounce(); 

    if(strcmp(mode, Default) == 0 )
    {
        SetCompAuto() ;   // comp set based on Max values   
    }
    else if(strcmp(mode, StandardFullDebug) == 0 )
    {
        SEventConfig::SetMaxRecord(maxbounce+1); 
        SEventConfig::SetMaxRec(maxbounce+1); 
        SEventConfig::SetMaxSeq(maxbounce+1); 
        SEventConfig::SetMaxPrd(maxbounce+1); 
        SEventConfig::SetMaxAux(maxbounce+1); 

        // since moved to compound sflat/stag so MaxFlat/MaxTag should now either be 0 or 1, nothing else  
        SEventConfig::SetMaxTag(1);   
        SEventConfig::SetMaxFlat(1); 
        SEventConfig::SetMaxSup(1); 

        SetCompAuto() ;   // comp set based on Max values   
    }
    else
    {
        LOG(fatal) << "mode [" << mode << "] IS NOT RECOGNIZED "  ;         
        LOG(fatal) << " options : " << Default << "," << StandardFullDebug ; 
        assert(0); 
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

    //meta->set_meta<unsigned>("CompMask", CompMask() );  
    //meta->set_meta<std::string>("CompMaskLabel", CompMaskLabel()); 

    meta->set_meta<unsigned>("GatherComp", GatherComp() );  
    meta->set_meta<unsigned>("SaveComp", SaveComp() );  

    meta->set_meta<std::string>("GatherCompLabel", GatherCompLabel()); 
    meta->set_meta<std::string>("SaveCompLabel", SaveCompLabel()); 

    meta->set_meta<float>("PropagateEpsilon", PropagateEpsilon() );  

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

