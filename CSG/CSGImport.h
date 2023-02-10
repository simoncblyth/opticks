#pragma once
/**
CSGImport.h 
==============

See::

    sysrap/tests/stree_load_test.sh 
    CSG/tests/CSGFoundry_importTree_Test.sh


**/

#include <string>
#include "plog/Severity.h"

struct stree ; 
struct snd ; 

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
    void importNames(); 

    void importSolid(); 
    CSGSolid* importRemainderSolid(int ridx, const char* rlabel); 
    CSGSolid* importFactorSolid(   int ridx, const char* rlabel); 
    CSGPrim*  importPrim( int primIdx, int lvid ); 

    void     importBinNode_r(const snd* nd, int idx); 
    CSGNode* importBinNode_v(const snd* nd, int idx); 

}; 



