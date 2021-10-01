#pragma once

/**
QTexMaker
==========

This struct exists instead of an additional QTex ctor
because of the need to create different texture types
with different template types so a single ctor 
would not be convenient as the signature excluding 
the return type needs to be distinctive. 

**/
struct NP ; 
struct float4 ; 
template <typename T> struct QTex ;  

#include "QUDARAP_API_EXPORT.hh"

struct QUDARAP_API QTexMaker
{
    static QTex<float4>* Make2d_f4( const NP* a, char filterMode ); 
};


