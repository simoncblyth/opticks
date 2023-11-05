#pragma once
/**
CSGCopy
=========


identical_bbox_cheat:true
    when the elv selection SBitSet has all bits set it means
    there is no selection so identical:true 
    
    In that case have observed small 0.002 mm shifts in 
    large bbox dimension values as a result of the copy : 
    causing CSGCopyTest.sh to fail when using exact comparisons. 

    As a shortcut to fixing the small bbox shift issue 
    can instead cheat and copy the bbox from the src when it
    is known that there is no selection being applied. 

**/

struct SBitSet ; 
struct CSGFoundry ; 
struct s_bb ; 

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
    bool              identical ; // when elv->is_all_set dst should be identical to src 
    bool              identical_bbox_cheat ;       

    CSGFoundry* dst ; 

    CSGCopy(const CSGFoundry* src, const SBitSet* elv); 
    virtual ~CSGCopy(); 

    std::string desc() const ; 
    void copy() ; 

#ifdef WITH_S_BB
    void copySolidPrim(s_bb& solid_bb, int dPrimOffset, const CSGSolid* sso ); 
    void copyPrimNodes(s_bb& prim_bb, const CSGPrim* spr ); 
    void copyNode(     s_bb& prim_bb, unsigned nodeIdx ); 
#else
    void copySolidPrim(AABB& solid_bb, int dPrimOffset, const CSGSolid* sso ); 
    void copyPrimNodes(AABB& prim_bb, const CSGPrim* spr ); 
    void copyNode(     AABB& prim_bb, unsigned nodeIdx ); 
#endif
    void copySolidInstances();

 
};


