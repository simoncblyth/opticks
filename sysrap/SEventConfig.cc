#include <sstream>
#include <cassert>

#include "SSys.hh"
#include "SEventConfig.hh"

int SEventConfig::_MaxGenstep = SSys::getenvint("OPTICKS_MAX_GENSTEP",  1000*K) ; 
int SEventConfig::_MaxPhoton  = SSys::getenvint("OPTICKS_MAX_PHOTON",      1*M) ; 
int SEventConfig::_MaxBounce  = SSys::getenvint("OPTICKS_MAX_BOUNCE",       9 ) ; 
int SEventConfig::_MaxRecord  = SSys::getenvint("OPTICKS_MAX_RECORD",      10 ) ; 

int SEventConfig::MaxGenstep(){  return _MaxGenstep ; }
int SEventConfig::MaxPhoton(){   return _MaxPhoton ; }
int SEventConfig::MaxBounce(){   return _MaxBounce ; }
int SEventConfig::MaxRecord(){   return _MaxRecord ; }

void SEventConfig::SetMaxGenstep(int max_genstep){ _MaxGenstep = max_genstep ; Check() ; }
void SEventConfig::SetMaxPhoton( int max_photon){  _MaxPhoton  = max_photon  ; Check() ; }
void SEventConfig::SetMaxBounce( int max_bounce){  _MaxBounce  = max_bounce  ; Check() ; }
void SEventConfig::SetMaxRecord( int max_record){  _MaxRecord  = max_record  ; Check() ; }

void SEventConfig::Check()
{
   assert( _MaxBounce >  0 && _MaxBounce <  16 ) ; 
   assert( _MaxRecord >= 0 && _MaxRecord <= 16 ) ; 
}

void SEventConfig::SetMax(int max_genstep_, int max_photon_, int max_bounce_, int max_record_ )
{ 
    SetMaxGenstep( max_genstep_ ); 
    SetMaxPhoton(  max_photon_  ); 
    SetMaxBounce(  max_bounce_  ); 
    SetMaxRecord(  max_record_  ); 
}
  
std::string SEventConfig::Desc()
{
    std::stringstream ss ; 
    ss << "SEventConfig::Desc"
       << " MaxGenstep " << MaxGenstep()
       << " MaxPhoton " << MaxPhoton()
       << " MaxBounce " << MaxBounce()
       << " MaxRecord " << MaxRecord()
       ;
    std::string s = ss.str(); 
    return s ; 
}


