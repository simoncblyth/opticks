#include <sstream>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "SSys.hh"
#include "SPath.hh"
#include "SEventConfig.hh"
#include "SRG.h"
#include "SComp.h"
#include "OpticksPhoton.hh"

#include "PLOG.hh"

const plog::Severity SEventConfig::LEVEL = PLOG::EnvLevel("SEventConfig", "DEBUG") ; 

const char* SEventConfig::_EventModeDefault = "Default" ; 
int SEventConfig::_MaxGenstepDefault = 1000*K ; 
int SEventConfig::_MaxBounceDefault = 9 ; 
int SEventConfig::_MaxRecordDefault = 0 ; 
int SEventConfig::_MaxRecDefault = 0 ; 
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

const char* SEventConfig::_CompMaskDefault = SComp::ALL_ ; 
float SEventConfig::_PropagateEpsilonDefault = 0.05f ; 
const char* SEventConfig::_InputPhotonDefault = nullptr ; 
const char* SEventConfig::_InputPhotonFrameDefault = nullptr ; 



const char* SEventConfig::_EventMode = SSys::getenvvar(kEventMode, _EventModeDefault ); 
int SEventConfig::_MaxGenstep   = SSys::getenvint(kMaxGenstep,  _MaxGenstepDefault ) ; 
int SEventConfig::_MaxPhoton    = SSys::getenvint(kMaxPhoton,   _MaxPhotonDefault ) ; 
int SEventConfig::_MaxSimtrace  = SSys::getenvint(kMaxSimtrace,   _MaxSimtraceDefault ) ; 
int SEventConfig::_MaxBounce    = SSys::getenvint(kMaxBounce, _MaxBounceDefault ) ; 
int SEventConfig::_MaxRecord    = SSys::getenvint(kMaxRecord, _MaxRecordDefault ) ;    
int SEventConfig::_MaxRec       = SSys::getenvint(kMaxRec, _MaxRecDefault ) ;   
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
unsigned SEventConfig::_CompMask  = SComp::Mask(SSys::getenvvar(kCompMask, _CompMaskDefault )) ;   
float SEventConfig::_PropagateEpsilon = SSys::getenvfloat(kPropagateEpsilon, _PropagateEpsilonDefault ) ; 
const char* SEventConfig::_InputPhoton = SSys::getenvvar(kInputPhoton, _InputPhotonDefault ); 
const char* SEventConfig::_InputPhotonFrame = SSys::getenvvar(kInputPhotonFrame, _InputPhotonFrameDefault ); 


const char* SEventConfig::EventMode(){ return _EventMode ; }
int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
int SEventConfig::MaxSimtrace(){   return _MaxSimtrace ; }
int SEventConfig::MaxBounce(){   return _MaxBounce ; }
int SEventConfig::MaxRecord(){   return _MaxRecord ; }
int SEventConfig::MaxRec(){      return _MaxRec ; }
int SEventConfig::MaxSeq(){      return _MaxSeq ; }
int SEventConfig::MaxPrd(){      return _MaxPrd ; }
int SEventConfig::MaxTag(){      return _MaxTag ; }
int SEventConfig::MaxFlat(){      return _MaxFlat ; }
float SEventConfig::MaxExtent(){ return _MaxExtent ; }
float SEventConfig::MaxTime(){   return _MaxTime ; }
const char* SEventConfig::OutFold(){   return _OutFold ; }
const char* SEventConfig::OutName(){   return _OutName ; }
int SEventConfig::RGMode(){  return _RGMode ; } 
unsigned SEventConfig::HitMask(){     return _HitMask ; }
unsigned SEventConfig::CompMask(){  return _CompMask; } 
float SEventConfig::PropagateEpsilon(){ return _PropagateEpsilon ; }
const char* SEventConfig::InputPhoton(){   return _InputPhoton ; }
const char* SEventConfig::InputPhotonFrame(){   return _InputPhotonFrame ; }


const char* SEventConfig::Default = "Default" ; 
const char* SEventConfig::StandardFullDebug = "StandardFullDebug" ; 

// considered calling Initialize from the below, but prefer to 
// have a fixed position G4CXOpticks::init where SEventConfig::Initialize is called
void SEventConfig::SetDefault(){            SetEventMode(Default)           ; } 
void SEventConfig::SetStandardFullDebug(){  SetEventMode(StandardFullDebug) ; }

void SEventConfig::SetEventMode(const char* mode){ _EventMode = mode ? strdup(mode) : nullptr ; Check() ; }
void SEventConfig::SetMaxGenstep(int max_genstep){ _MaxGenstep = max_genstep ; Check() ; }
void SEventConfig::SetMaxPhoton( int max_photon){  _MaxPhoton  = max_photon  ; Check() ; }
void SEventConfig::SetMaxSimtrace( int max_simtrace){  _MaxSimtrace  = max_simtrace  ; Check() ; }
void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
void SEventConfig::SetMaxRecord( int max_record){  _MaxRecord  = max_record  ; Check() ; }
void SEventConfig::SetMaxRec(    int max_rec){     _MaxRec     = max_rec     ; Check() ; }
void SEventConfig::SetMaxSeq(    int max_seq){     _MaxSeq     = max_seq     ; Check() ; }
void SEventConfig::SetMaxPrd(    int max_prd){     _MaxPrd     = max_prd     ; Check() ; }
void SEventConfig::SetMaxTag(    int max_tag){     _MaxTag     = max_tag     ; Check() ; }
void SEventConfig::SetMaxFlat(    int max_flat){     _MaxFlat     = max_flat     ; Check() ; }
void SEventConfig::SetMaxExtent( float max_extent){ _MaxExtent = max_extent  ; Check() ; }
void SEventConfig::SetMaxTime(   float max_time){   _MaxTime = max_time  ; Check() ; }
void SEventConfig::SetOutFold(   const char* outfold){   _OutFold = outfold ? strdup(outfold) : nullptr ; Check() ; }
void SEventConfig::SetOutName(   const char* outname){   _OutName = outname ? strdup(outname) : nullptr ; Check() ; }
void SEventConfig::SetHitMask(   const char* abrseq, char delim){  _HitMask = OpticksPhoton::GetHitMask(abrseq,delim) ; }

void SEventConfig::SetRGMode(   const char* rg_mode){   _RGMode = SRG::Type(rg_mode) ; Check() ; }
void SEventConfig::SetRGModeSimulate(){  SetRGMode( SRG::SIMULATE_ ); }
void SEventConfig::SetRGModeSimtrace(){  SetRGMode( SRG::SIMTRACE_ ); }
void SEventConfig::SetRGModeRender(){    SetRGMode( SRG::RENDER_ ); }

void SEventConfig::SetPropagateEpsilon(float eps){ _PropagateEpsilon = eps ; Check() ; }
void SEventConfig::SetInputPhoton(const char* ip){   _InputPhoton = ip ? strdup(ip) : nullptr ; Check() ; }
void SEventConfig::SetInputPhotonFrame(const char* ip){   _InputPhotonFrame = ip ? strdup(ip) : nullptr ; Check() ; }

void SEventConfig::SetCompMask_(unsigned mask){ _CompMask = mask ; }
void SEventConfig::SetCompMask(const char* names, char delim){  SetCompMask_( SComp::Mask(names,delim)) ; }
void SEventConfig::SetCompMaskAuto(){ SetCompMask_( CompMaskAuto() ) ; }


/**
SEventConfig::CompMaskAuto
---------------------------

**/

unsigned SEventConfig::CompMaskAuto()
{
    unsigned mask = 0 ;     
    if(IsRGModeSimulate())
    {
        if(MaxGenstep()>0)   mask |= SCOMP_GENSTEP ; 
        if(MaxPhoton()>0)
        {
            mask |= SCOMP_PHOTON ;  
            mask |= SCOMP_HIT ; 
            //mask |= SCOMP_SEED ;   // only needed for deep debugging 
        }
        if(MaxRecord()>0)    mask |= SCOMP_RECORD ; 
        if(MaxRec()>0)       mask |= SCOMP_REC ; 
        if(MaxSeq()>0)       mask |= SCOMP_SEQ ; 
        if(MaxPrd()>0)       mask |= SCOMP_PRD ; 
        if(MaxTag()>0)       mask |= SCOMP_TAG ; 
        if(MaxFlat()>0)      mask |= SCOMP_FLAT ; 
    }
    else if(IsRGModeSimtrace())
    {
        if(MaxGenstep()>0)   mask |= SCOMP_GENSTEP ; 
        if(MaxSimtrace()>0)  mask |= SCOMP_SIMTRACE ; 
    } 
    else if(IsRGModeRender())
    {
        mask |= SCOMP_PIXEL ; 
    }
    return mask ; 
}







bool SEventConfig::IsRGModeRender(){   return RGMode() == SRG_RENDER   ; } 
bool SEventConfig::IsRGModeSimtrace(){ return RGMode() == SRG_SIMTRACE ; } 
bool SEventConfig::IsRGModeSimulate(){ return RGMode() == SRG_SIMULATE ; } 

const char* SEventConfig::RGModeLabel(){ return SRG::Name(_RGMode) ; }
std::string SEventConfig::HitMaskLabel(){  return OpticksPhoton::FlagMask( _HitMask ) ; }
std::string SEventConfig::CompMaskLabel(){ return SComp::Desc( _CompMask ) ; }

void SEventConfig::CompList( std::vector<unsigned>& comps )
{
    SComp::CompListMask(comps, CompMask() ); 
}  


void SEventConfig::Check()
{
   assert( _MaxBounce >  0 && _MaxBounce <  16 ) ; 
   assert( _MaxRecord >= 0 && _MaxRecord <= 16 ) ; 
   assert( _MaxRec    >= 0 && _MaxRec    <= 16 ) ; 
   assert( _MaxSeq    >= 0 && _MaxSeq    <= 16 ) ; 
   assert( _MaxPrd    >= 0 && _MaxPrd    <= 16 ) ; 
   assert( _MaxTag    >= 0 && _MaxTag    <= 24 ) ; 
   assert( _MaxFlat    >= 0 && _MaxFlat    <= 24 ) ; 
}

 
std::string SEventConfig::Desc()
{
    std::stringstream ss ; 
    ss << "SEventConfig::Desc" << std::endl 
       << std::setw(25) << kEventMode
       << std::setw(20) << " EventMode " << " : " << EventMode() << std::endl 
       << std::setw(25) << kMaxGenstep 
       << std::setw(20) << " MaxGenstep " << " : " << MaxGenstep() << std::endl 
       << std::setw(25) << kMaxPhoton 
       << std::setw(20) << " MaxPhoton " << " : " << MaxPhoton() << std::endl 
       << std::setw(25) << kMaxSimtrace 
       << std::setw(20) << " MaxSimtrace " << " : " << MaxSimtrace() << std::endl 
       << std::setw(25) << kMaxBounce
       << std::setw(20) << " MaxBounce " << " : " << MaxBounce() << std::endl 
       << std::setw(25) << kMaxRecord
       << std::setw(20) << " MaxRecord " << " : " << MaxRecord() << std::endl 
       << std::setw(25) << kMaxRec
       << std::setw(20) << " MaxRec " << " : " << MaxRec() << std::endl 
       << std::setw(25) << kMaxSeq
       << std::setw(20) << " MaxSeq " << " : " << MaxSeq() << std::endl 
       << std::setw(25) << kMaxPrd
       << std::setw(20) << " MaxPrd " << " : " << MaxPrd() << std::endl 
       << std::setw(25) << kMaxTag
       << std::setw(20) << " MaxTag " << " : " << MaxTag() << std::endl 
       << std::setw(25) << kMaxFlat
       << std::setw(20) << " MaxFlat " << " : " << MaxFlat() << std::endl 
       << std::setw(25) << kHitMask
       << std::setw(20) << " HitMask " << " : " << HitMask() << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " HitMaskLabel " << " : " << HitMaskLabel() << std::endl 
       << std::setw(25) << kMaxExtent
       << std::setw(20) << " MaxExtent " << " : " << MaxExtent() << std::endl 
       << std::setw(25) << kMaxTime
       << std::setw(20) << " MaxTime " << " : " << MaxTime() << std::endl 
       << std::setw(25) << kRGMode
       << std::setw(20) << " RGMode " << " : " << RGMode() << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " RGModeLabel " << " : " << RGModeLabel() << std::endl 
       << std::setw(25) << kCompMask
       << std::setw(20) << " CompMask " << " : " << CompMask() << std::endl 
       << std::setw(25) << ""
       << std::setw(20) << " CompMaskLabel " << " : " << CompMaskLabel() << std::endl 

       << std::setw(25) << kOutFold
       << std::setw(20) << " OutFold " << " : " << OutFold() << std::endl 
       << std::setw(25) << kOutName
       << std::setw(20) << " OutName " << " : " << ( OutName() ? OutName() : "-" )  << std::endl 
       << std::setw(25) << kPropagateEpsilon
       << std::setw(20) << " PropagateEpsilon " << " : " << std::fixed << std::setw(10) << std::setprecision(4) << PropagateEpsilon() << std::endl 
       << std::setw(25) << kInputPhoton
       << std::setw(20) << " InputPhoton " << " : " << ( InputPhoton() ? InputPhoton() : "-" )  << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}


/**
SEventConfig::OutDir SEventConfig::OutPath
--------------------------------------------

Q: Are these methods used ? 
A: YES by CSGOptiX::render_snap

::

    const char* outpath = SEventConfig::OutPath(name, -1, ".jpg" );


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

HMM: if change the max after calling this need to SEventConfig::SetCompMaskAuto() 

**/


void SEventConfig::Initialize() // static
{
    const char* mode = EventMode(); 
    int maxbounce = MaxBounce(); 

    if(strcmp(mode, Default) == 0 )
    {
        SetCompMaskAuto() ;   // comp set based on Max values   
    }
    else if(strcmp(mode, StandardFullDebug) == 0 )
    {
        SEventConfig::SetMaxRecord(maxbounce+1); 
        SEventConfig::SetMaxRec(maxbounce+1); 
        SEventConfig::SetMaxSeq(maxbounce+1); 
        SEventConfig::SetMaxPrd(maxbounce+1); 

        unsigned slots = 24 ;  // HMM: stag::SLOTS  
        SEventConfig::SetMaxTag(slots);             // stag::NSEQ*(64/stag::BITS) = 2*12 = 24
        SEventConfig::SetMaxFlat(slots);             // stag::NSEQ*(64/stag::BITS) = 2*12 = 24
        SetCompMaskAuto() ;   // comp set based on Max values   
    }
    else
    {
        LOG(fatal) << "mode [" << mode << "] IS NOT RECOGNIZED "  ;         
        LOG(fatal) << " options : " << Default << "," << StandardFullDebug ; 
        assert(0); 
    }
}


