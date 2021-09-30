#pragma once

struct NP ; 
struct float4 ; 
template <typename T> struct QTex ;  

#include "QUDARAP_API_EXPORT.hh"

struct QUDARAP_API QTexMaker
{
    static QTex<float4>* Make2d_f4( const NP* a, char filterMode ); 
};


