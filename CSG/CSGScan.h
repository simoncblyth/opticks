#pragma once
/**
CSGScan.h : CPU testing of GPU csg_intersect impl
==================================================


**/

#include <string>
#include <vector>
#include "squad.h"
#include "NPFold.h"

#include "CSG_API_EXPORT.hh"

struct CSGFoundry ; 
struct CSGSolid ; 

struct CSG_API CSGScan
{
    CSGScan( const CSGFoundry* fd_, const CSGSolid* solid_, const char* opt );   

    void add_scan(const char* opt);
    void add_circle_scan(); 
    void add_rectangle_scan(); 
    void _add_rectangle_scan(float t_min, unsigned n, float halfside, float y ) ;
    void add_axis_scan(); 


    void add_q(const float t_min, const float3& ray_origin, const float3& ray_direction );
    void add_q(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs );

    void add_isect( const float4& i, bool valid_isect, const quad4& q ); 

    void intersect_prim_scan();


    std::string brief() const ;
    void dump( const quad4& rec ); 

    NPFold* serialize() const ; 
    void save(const char* base, const char* sub) const ;


    const CSGFoundry* fd ; 
    const CSGPrim*    prim0 ; 
    const CSGNode*    node0 ; 
    const float4*     plan0 ; 
    const qat4*       itra0 ; 
    const CSGSolid*   so ;
 
    int primIdx0 ; 
    int primIdx1 ; 
    int primIdx ; 
    const CSGPrim* prim ; 
    int nodeOffset ; 
    const CSGNode* node ; 


    std::vector<quad4> qq ; 
    std::vector<quad4> tt ;
 
};



