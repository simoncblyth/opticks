#pragma once

struct float4 ; 
struct float3 ; 
struct NP ; 

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGGrid
{
    static const char* BASE ; 

    float4 ce ;
    float margin ; 
    int nx ; 
    int ny ; 
    int nz ; 
    float3 gridscale ; 
    int ni ; 
    int nj ; 
    int nk ; 
    NP* sdf ; 
    float* sdf_v ; 
    NP* xyzd ; 
    float* xyzd_v ; 

    CSGGrid( const float4& ce_, int nx_, int ny_, int nz_ ); 

    void init(); 
    void init_meta(); 
    void scan( std::function<float(const float3&)> sdf  ) ; 

    void save(const char* geom, const char* base=nullptr) const ; 
}; 


