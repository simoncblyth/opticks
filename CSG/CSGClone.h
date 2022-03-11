#pragma once

struct CSGFoundry ; 
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGClone
{
    static CSGFoundry* Clone(const CSGFoundry* src ); 
    static void Copy(CSGFoundry* dst, const CSGFoundry* src ) ; 


};
