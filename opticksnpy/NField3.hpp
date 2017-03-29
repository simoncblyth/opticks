#pragma once

#include <functional>
#include "NQuad.hpp"

#include "NPY_API_EXPORT.hh"



struct NPY_API NField3
{
    static const int ZCORNER = 8 ; 
    static const nvec3 ZOFFSETS[ZCORNER] ;

    typedef std::function<float(float,float,float)> FN ;  
    NField3( FN* f, const nvec3& min, const nvec3& max);
    std::string desc();

    nvec3 position( const nvec3& fpos ) const;         // fractional position in 0:1 to world position in min:max

    float operator()( const nvec3& fpos ) const;  // fractional position in 0:1 to field value

    int zcorners( const nvec3& fpos, float fdelta ) const ;

    FN*          f ; 

    nvec3       min  ;
    nvec3       max  ;
    nvec3       side ; 


};


