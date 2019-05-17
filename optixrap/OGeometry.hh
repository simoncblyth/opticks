#pragma once

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "OXRAP_API_EXPORT.hh"

struct OXRAP_API  OGeometry 
{
    optix::Geometry           g ; 
#if OPTIX_VERSION >= 60000
    optix::GeometryTriangles  gt ; 
#endif
    bool isGeometry() const ;  
    bool isGeometryTriangles() const ;  
};



