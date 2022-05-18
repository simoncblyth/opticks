#include <sstream>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "SSys.hh"
#include "SPath.hh"
#include "SEventConfig.hh"
#include "SRG.h"
#include "OpticksPhoton.hh"

#ifdef __APPLE__
int  SEventConfig::_MaxPhotonDefault = 1*M ; 
#else
int  SEventConfig::_MaxPhotonDefault = 3*M ; 
#endif

int SEventConfig::_MaxGenstep = SSys::getenvint(kMaxGenstep,  1000*K) ; 
int SEventConfig::_MaxPhoton  = SSys::getenvint(kMaxPhoton,   _MaxPhotonDefault ) ; 
int SEventConfig::_MaxBounce  = SSys::getenvint(kMaxBounce, 9 ) ; 
int SEventConfig::_MaxRecord  = SSys::getenvint(kMaxRecord, 0 ) ;    // full step record
int SEventConfig::_MaxRec     = SSys::getenvint(kMaxRec, 0 ) ;    // compressed record  
int SEventConfig::_MaxSeq     = SSys::getenvint(kMaxSeq,          0 ) ;    // compressed record  
float SEventConfig::_MaxExtent = SSys::getenvfloat(kMaxExtent,  1000.f );  // mm 
float SEventConfig::_MaxTime   = SSys::getenvfloat(kMaxTime,    10.f );    // ns
const char* SEventConfig::_OutFold = SSys::getenvvar(kOutFold,  "$TMP" ); 
const char* SEventConfig::_OutName = SSys::getenvvar(kOutName,  nullptr ); 
int SEventConfig::_RGMode = SRG::Type(SSys::getenvvar(kRGMode, "simulate")) ;    
unsigned SEventConfig::_HitMask  = OpticksPhoton::GetHitMask(SSys::getenvvar(kHitMask, "SD" )) ;   

int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
int SEventConfig::MaxBounce(){   return _MaxBounce ; }
int SEventConfig::MaxRecord(){   return _MaxRecord ; }
int SEventConfig::MaxRec(){      return _MaxRec ; }
int SEventConfig::MaxSeq(){      return _MaxSeq ; }
float SEventConfig::MaxExtent(){ return _MaxExtent ; }
float SEventConfig::MaxTime(){   return _MaxTime ; }
const char* SEventConfig::OutFold(){   return _OutFold ; }
const char* SEventConfig::OutName(){   return _OutName ; }
int SEventConfig::RGMode(){  return _RGMode ; } 

bool SEventConfig::IsRGModeRender(){   return RGMode() == SRG_RENDER   ; } 
bool SEventConfig::IsRGModeSimtrace(){ return RGMode() == SRG_SIMTRACE ; } 
bool SEventConfig::IsRGModeSimulate(){ return RGMode() == SRG_SIMULATE ; } 


unsigned SEventConfig::HitMask(){     return _HitMask ; }

void SEventConfig::SetMaxGenstep(int max_genstep){ _MaxGenstep = max_genstep ; Check() ; }
void SEventConfig::SetMaxPhoton( int max_photon){  _MaxPhoton  = max_photon  ; Check() ; }
void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
void SEventConfig::SetMaxRecord( int max_record){  _MaxRecord  = max_record  ; Check() ; }
void SEventConfig::SetMaxRec(    int max_rec){     _MaxRec     = max_rec     ; Check() ; }
void SEventConfig::SetMaxSeq(    int max_seq){     _MaxSeq     = max_seq     ; Check() ; }
void SEventConfig::SetMaxExtent( float max_extent){ _MaxExtent = max_extent  ; Check() ; }
void SEventConfig::SetMaxTime(   float max_time){   _MaxTime = max_time  ; Check() ; }
void SEventConfig::SetOutFold(   const char* outfold){   _OutFold = outfold ? strdup(outfold) : nullptr ; Check() ; }
void SEventConfig::SetOutName(   const char* outname){   _OutName = outname ? strdup(outname) : nullptr ; Check() ; }
void SEventConfig::SetRGMode(   const char* rg_mode){   _RGMode = SRG::Type(rg_mode) ; Check() ; }
void SEventConfig::SetHitMask(const char* abrseq, char delim){  _HitMask = OpticksPhoton::GetHitMask(abrseq,delim) ; }

const char* SEventConfig::RGModeLabel(){ return SRG::Name(_RGMode) ; }
std::string SEventConfig::HitMaskLabel(){  return OpticksPhoton::FlagMask( _HitMask ) ; }


void SEventConfig::Check()
{
   assert( _MaxBounce >  0 && _MaxBounce <  16 ) ; 
   assert( _MaxRecord >= 0 && _MaxRecord <= 16 ) ; 
   assert( _MaxRec    >= 0 && _MaxRec    <= 16 ) ; 
   assert( _MaxSeq    >= 0 && _MaxSeq    <= 16 ) ; 
}

void SEventConfig::SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_, int max_rec_, int max_seq_ )
{ 
    SetMaxGenstep( max_genstep_ ); 
    SetMaxPhoton(  max_photon_  ); 
    SetMaxBounce(  max_bounce_  ); 
    SetMaxRecord(  max_record_  ); 
    SetMaxRec(     max_rec_  ); 
    SetMaxSeq(     max_seq_  ); 
}
  
std::string SEventConfig::Desc()
{
    std::stringstream ss ; 
    ss << "SEventConfig::Desc" << std::endl 
       << std::setw(25) << kMaxGenstep 
       << std::setw(20) << " MaxGenstep " << " : " << MaxGenstep() << std::endl 
       << std::setw(25) << kMaxPhoton 
       << std::setw(20) << " MaxPhoton " << " : " << MaxPhoton() << std::endl 
       << std::setw(25) << kMaxBounce
       << std::setw(20) << " MaxBounce " << " : " << MaxBounce() << std::endl 
       << std::setw(25) << kMaxRecord
       << std::setw(20) << " MaxRecord " << " : " << MaxRecord() << std::endl 
       << std::setw(25) << kMaxRec
       << std::setw(20) << " MaxRec " << " : " << MaxRec() << std::endl 
       << std::setw(25) << kMaxSeq
       << std::setw(20) << " MaxSeq " << " : " << MaxSeq() << std::endl 
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
       << std::setw(25) << kOutFold
       << std::setw(20) << " OutFold " << " : " << OutFold() << std::endl 
       << std::setw(25) << kOutName
       << std::setw(20) << " OutName " << " : " << ( OutName() ? OutName() : "-" )  << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}



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




