#pragma once
/**
CSGGrid.h : signed distance field grid 
=======================================

* NB: no other CSG dependency, can and should be relocated down to sysrap if still needed 
* instance returned from CSGQuery::scanPrim which is used by CSGGeometry::saveSignedDistanceField

::

    epsilon:GeoChain blyth$ opticks-f CSGGrid.h 
    ./CSG/CSGGeometry.cc:#include "CSGGrid.h"
    ./CSG/CSGGrid.cc:#include "CSGGrid.h"
    ./CSG/CSGQuery.cc:#include "CSGGrid.h"
    ./CSG/CMakeLists.txt:    CSGGrid.h
    ./CSG/CSGGrid.h:CSGGrid.h : signed distance field grid 
    epsilon:opticks blyth$ 


**/

#include <functional>
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


