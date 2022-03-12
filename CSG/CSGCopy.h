#pragma once

struct SBitSet ; 
struct CSGFoundry ; 

#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGCopy
{
    static const plog::Severity LEVEL ; 
    static CSGFoundry* Clone( const CSGFoundry* src ); 
    static CSGFoundry* Select(const CSGFoundry* src, const SBitSet* elv ); 

    static void Copy(CSGFoundry* dst, const CSGFoundry* src, const SBitSet* elv ) ; 
};
