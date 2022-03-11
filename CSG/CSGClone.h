#pragma once

struct CSGFoundry ; 
#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGClone
{
    static const plog::Severity LEVEL ; 
    static CSGFoundry* Clone(const CSGFoundry* src ); 
    static void Copy(CSGFoundry* dst, const CSGFoundry* src ) ; 
};
