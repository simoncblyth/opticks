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
#include "plog/Severity.h"

#include "QUDARAP_API_EXPORT.hh"

struct QUDARAP_API QTexMaker
{
    static const plog::Severity LEVEL ; 
    static QTex<float4>* Make2d_f4( const NP* icdf, char filterMode ); 
    static QTex<float4>* Make2d_f4_( const NP* a, char filterMode ); 
};


