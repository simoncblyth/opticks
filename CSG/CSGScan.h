#pragma once

#include <string>
#include <vector>
#include "squad.h"

#include "CSG_API_EXPORT.hh"

struct CSGFoundry ; 
struct CSGSolid ; 

struct CSG_API CSGScan
{
    CSGScan( const char* dir_, const CSGFoundry* foundry_, const CSGSolid* solid_ );   

    void trace(const float t_min, const float3& ray_origin, const float3& ray_direction );
    void trace(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs );

    void record(bool valid_isect, const float4& isect,  const float3& ray_origin, const float3& ray_direction ) ;

    void circle_scan(); 
    void axis_scan(); 
    void rectangle_scan(); 
    void _rectangle_scan(float t_min, unsigned n, float halfside, float y ) ;

    std::string brief() const ;
    void dump( const quad4& rec ); 
    void save(const char* sub);


    const char*    dir ; 
    const CSGFoundry* foundry ; 
    const CSGPrim*    prim0 ; 
    const CSGNode*    node0 ; 
    const float4*  plan0 ; 
    const qat4*    itra0 ; 

    const CSGSolid* solid ; 
    unsigned primIdx0 ; 
    unsigned primIdx1 ; 


    std::vector<quad4> recs ; 
};



