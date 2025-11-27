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
struct CSGParams ; 

struct CSG_API CSGScan
{
    CSGScan( const CSGFoundry* fd_, const CSGSolid* solid_, const char* opt );   

    void initGeom_h(); 
    void initRays_h(const char* opts_); 

#ifdef WITH_CUDA
    void initGeom_d(); 
    void initRays_d(); 
    void initParams_d(); 
#endif

    void add_scan(std::vector<quad4>& qq, const char* opt);
    void add_axis_scan(std::vector<quad4>& qq); 
    void add_circle_scan(std::vector<quad4>& qq); 
    void add_rectangle_scan(std::vector<quad4>& qq); 
    void _add_rectangle_scan(std::vector<quad4>& qq, float t_min, unsigned n, float halfside, float y ) ;

    void add_q(std::vector<quad4>& qq, const float t_min, const float3& ray_origin, const float3& ray_direction );
    void add_q(std::vector<quad4>& qq, const float t_min, const float3& ray_origin, const std::vector<float3>& dirs );

    void intersect_h();

#ifdef WITH_CUDA
    void intersect_d();
    void download(); 
#endif

    std::string brief() const ;
    std::string brief(CSGParams* s) const ;
    void dump( const quad4& rec ); 

    NPFold* serialize_(CSGParams* s) const ; 
    NPFold* serialize() const ; 
    void save(const char* base) const ;

    const CSGFoundry* fd ; 
    const CSGPrim*    prim0 ; 
    const CSGNode*    node0 ; 
    const CSGSolid*   so ;
 
    int primIdx0 ; 
    int primIdx1 ; 
    int primIdx ; 
    const CSGPrim* prim ; 
    int nodeOffset ; 
    const CSGNode* node ; 

    CSGParams* h ;
    CSGParams* d ;
    CSGParams* d_d ;
    CSGParams* c ;

 
};



