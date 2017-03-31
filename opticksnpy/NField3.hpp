#pragma once

#include <functional>
#include "NQuad.hpp"

#include "NPY_API_EXPORT.hh"



template<typename FVec, typename IVec, int DIM>
struct NPY_API NField
{
    static const int ZCORNER = 8 ; 
    static const FVec ZOFFSETS[ZCORNER] ;

    typedef std::function<float(float,float,float)> FN ;  
    NField( FN* f, const FVec& min, const FVec& max);
    std::string desc();

    FVec position( const FVec& fpos ) const;         // fractional position in 0:1 to world position in min:max

    float operator()( const FVec& fpos ) const;  // fractional position in 0:1 to field value

    int zcorners( const FVec& fpos, float fdelta ) const ;

    FN*  f ; 
    FVec min  ;
    FVec max  ;
    FVec side ; 


};


