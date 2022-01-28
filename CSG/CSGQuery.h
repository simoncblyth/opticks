#pragma once

struct CSGFoundry ; 
struct CSGPrim ; 
struct CSGNode ; 
struct float4 ; 
struct float3 ; 
struct quad4 ; 
struct qat4 ; 
struct CSGGrid ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGQuery 
{
    CSGQuery(const CSGFoundry* fd); 

    void     init(); 
    void     selectPrim(unsigned solidIdx, unsigned primIdxRel );
    void     selectPrim(const CSGPrim* pr );

    void     dumpPrim() const ;
    CSGGrid* scanPrim(int resolution) const ;
    float operator()(const float3& position) const ;

    bool intersect( quad4& isect,  float t_min, const quad4& p ) const ;
    bool intersect( quad4& isect,  float t_min, const float3& ray_origin, const float3& ray_direction ) const ;


    const CSGFoundry* fd ; 
    const CSGPrim* prim0 ; 
    const CSGNode* node0 ; 
    const float4*  plan0 ; 
    const qat4*    itra0 ; 

    const CSGPrim* select_prim ;
    int            select_numNode ;
    const CSGNode* select_root ;
 

};


