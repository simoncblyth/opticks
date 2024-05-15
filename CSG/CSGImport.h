#pragma once
/**
CSGImport.h 
==============

See::

    sysrap/tests/stree_load_test.sh 
    CSG/tests/CSGFoundry_importTree_Test.sh


CAUTION : SOME PARALLEL DEV NEEDS REVIEW, CONSOLIDATION::

    CSG_stree_Convert.h
    CSG_stree_Convert_test.cc
    CSG_stree_Convert_test.sh


**/

#include <string>
#include "plog/Severity.h"

struct stree ; 
struct snode ; 
struct sn ; 

struct CSGFoundry ; 
struct CSGSolid ; 
struct CSGPrim ; 
struct CSGNode ; 

#include "CSG_API_EXPORT.hh"


struct CSG_API CSGImport  // HMM: maybe CSGCreate is a better name ? 
{
    static const plog::Severity LEVEL ; 
    static const int LVID ; 
    static const int NDID ; 

    CSGFoundry*  fd ; 
    const stree* st ; 

    CSGImport( CSGFoundry* fd );  
 
    void import(); 
    void importNames(); 
    void importSolid(); 
    void importInst(); 

    CSGSolid* importSolidRemainder(int ridx, const char* rlabel); 
    CSGSolid* importSolidFactor(   int ridx, const char* rlabel); 

    CSGPrim*  importPrim( int primIdx, const snode& node ); 
    CSGNode*  importNode( int nodeOffset, int partIdx, const snode& node, const sn* nd); 
    CSGNode*  importListnode(int nodeOffset, int partIdx, const snode& node, const sn* nd); 

}; 



