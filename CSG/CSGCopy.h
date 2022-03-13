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
    static void CopySolidPrim(int dPrimOffset, CSGFoundry* dst, const CSGSolid* sso, const CSGFoundry* src, const SBitSet* elv, bool dump ); 
    static void CopyPrimNodes(AABB& bb, CSGFoundry* dst, const CSGPrim* spr, const CSGFoundry* src, bool dump ); 
    static void CopySolidInstances( const int* solidMap, unsigned sNumSolid, CSGFoundry* dst, const CSGFoundry* src ); 
};


