#pragma once

struct CSGFoundry ; 
#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

template <unsigned N> struct SEnabled ; 


struct CSG_API CSGSelection
{
    static const plog::Severity LEVEL ; 
    static CSGFoundry* Apply( const CSGFoundry* src, const SEnabled<64>* mmidx, const SEnabled<512>* lvidx ); 
}; 


