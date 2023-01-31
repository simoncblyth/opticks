#pragma once
/**
CSGImport.h 
======================================================

**/

#include <string>

#include "plog/Severity.h"

struct stree ; 
struct snode ; 

struct CSGFoundry ; 
struct CSGSolid ; 
struct CSGPrim ; 
struct CSGNode ; 

#include "CSG_API_EXPORT.hh"


struct CSG_API CSGImport
{
    static const plog::Severity LEVEL ; 

    CSGFoundry*  fd ; 
    const stree* st ; 

    CSGImport( CSGFoundry* fd );  
 
    void importTree(const stree* st); 

    void importSolid(); 
    CSGSolid* importRemainderSolid(int ridx, const char* rlabel); 
    CSGSolid* importFactorSolid(   int ridx, const char* rlabel); 
    CSGPrim*  importPrim( int primIdx, const snode& node ); 

}; 




