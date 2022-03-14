#pragma once

struct SBitSet ; 
struct CSGFoundry ; 

#include "plog/Severity.h"
#include "CSG_API_EXPORT.hh"

struct CSG_API CSGCopy
{
    static const plog::Severity LEVEL ; 
    static const int DUMP_RIDX ; 
    static const int DUMP_NPS ; // 3-bits bitfield (node,prim,solid)  7:111 6:110 5:101 4:100 3:011 2:010 1:001 0:000 

    static unsigned Dump( unsigned sSolidIdx ); 
    static CSGFoundry* Clone( const CSGFoundry* src ); 
    static CSGFoundry* Select(const CSGFoundry* src, const SBitSet* elv ); 

    const CSGFoundry* src ; 
    unsigned          sNumSolid ; 
    int*              solidMap ;  
    unsigned          sSolidIdx ; 
    const SBitSet*    elv ; 

    CSGFoundry* dst ; 

    CSGCopy(const CSGFoundry* src, const SBitSet* elv); 
    virtual ~CSGCopy(); 

    std::string desc() const ; 
    void copy() ; 
    void copyMeshName(); 
    void copySolidPrim(AABB& solid_bb, int dPrimOffset, const CSGSolid* sso ); 
    void copyPrimNodes(AABB& prim_bb, const CSGPrim* spr ); 
    void copyNode(     AABB& prim_bb, unsigned nodeIdx ); 
    void copySolidInstances();

 
};


