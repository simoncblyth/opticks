#include <sstream>
#include <cassert>

#include "SSys.hh"
#include "SEventConfig.hh"
#include "OpticksPhoton.hh"


int SEventConfig::_MaxGenstep = SSys::getenvint("OPTICKS_MAX_GENSTEP",  1000*K) ; 

#ifdef __APPLE__
int SEventConfig::_MaxPhoton  = SSys::getenvint("OPTICKS_MAX_PHOTON",      1*M) ; 
#else
int SEventConfig::_MaxPhoton  = SSys::getenvint("OPTICKS_MAX_PHOTON",      3*M) ; 
#endif


int SEventConfig::_MaxBounce  = SSys::getenvint("OPTICKS_MAX_BOUNCE",       9 ) ; 
int SEventConfig::_MaxRecord  = SSys::getenvint("OPTICKS_MAX_RECORD",       0 ) ;    // full step record
int SEventConfig::_MaxRec     = SSys::getenvint("OPTICKS_MAX_REC",          0 ) ;    // compressed record  
float SEventConfig::_MaxExtent = SSys::getenvfloat("OPTICKS_MAX_EXTENT",  1000.f );  // mm 
float SEventConfig::_MaxTime   = SSys::getenvfloat("OPTICKS_MAX_TIME",    10.f );    // ns


int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
int SEventConfig::MaxBounce(){   return _MaxBounce ; }
int SEventConfig::MaxRecord(){   return _MaxRecord ; }
int SEventConfig::MaxRec(){      return _MaxRec ; }
float SEventConfig::MaxExtent(){ return _MaxExtent ; }
float SEventConfig::MaxTime(){   return _MaxTime ; }

void SEventConfig::SetMaxGenstep(int max_genstep){ _MaxGenstep = max_genstep ; Check() ; }
void SEventConfig::SetMaxPhoton( int max_photon){  _MaxPhoton  = max_photon  ; Check() ; }
void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
void SEventConfig::SetMaxRecord( int max_record){  _MaxRecord  = max_record  ; Check() ; }
void SEventConfig::SetMaxRec(    int max_rec){     _MaxRec     = max_rec     ; Check() ; }
void SEventConfig::SetMaxExtent( float max_extent){ _MaxExtent = max_extent  ; Check() ; }
void SEventConfig::SetMaxTime(   float max_time){   _MaxTime = max_time  ; Check() ; }


unsigned SEventConfig::_HitMask  = OpticksPhoton::GetHitMask(SSys::getenvvar("OPTICKS_HITMASK", "SD" )) ;   
unsigned SEventConfig::HitMask(){     return _HitMask ; }
void SEventConfig::SetHitMask(const char* abrseq, char delim){  _HitMask = OpticksPhoton::GetHitMask(abrseq,delim) ; }
std::string SEventConfig::HitMaskDesc(){  return OpticksPhoton::FlagMask( _HitMask ) ; }


void SEventConfig::Check()
{
   assert( _MaxBounce >  0 && _MaxBounce <  16 ) ; 
   assert( _MaxRecord >= 0 && _MaxRecord <= 16 ) ; 
   assert( _MaxRec    >= 0 && _MaxRec    <= 16 ) ; 
}

void SEventConfig::SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_, int max_rec_ )
{ 
    SetMaxGenstep( max_genstep_ ); 
    SetMaxPhoton(  max_photon_  ); 
    SetMaxBounce(  max_bounce_  ); 
    SetMaxRecord(  max_record_  ); 
    SetMaxRec(     max_rec_  ); 
}
  
std::string SEventConfig::Desc()
{
    std::stringstream ss ; 
    ss << "SEventConfig::Desc"
       << " MaxGenstep " << MaxGenstep()
       << " MaxPhoton " << MaxPhoton()
       << " MaxBounce " << MaxBounce()
       << " MaxRecord(full) " << MaxRecord()
       << " MaxRec(compressed) " << MaxRec()
       << " HitMask " << HitMask()
       << " HitMaskDesc " << HitMaskDesc()
       << " MaxExtent " << MaxExtent()
       << " MaxTime " << MaxTime()
       ;
    std::string s = ss.str(); 
    return s ; 
}


