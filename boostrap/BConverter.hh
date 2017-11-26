
#pragma once

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API BConverter {
  public:
     template<typename T, typename S> 
     static T round_to_even(const S& x) ;

     static short shortnorm( float v, float center, float extent ); 
     static unsigned char my__float2uint_rn( float fv ) ;
    
     static short shortnorm_old( float v, float center, float extent ); 
     static unsigned char my__float2uint_rn_old( float fv ) ;


};

#include "BRAP_TAIL.hh"


