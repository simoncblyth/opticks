#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "OptiXGeometry.hh"
#include "GPropertyMap.hh" 

class GGeo ; 
class GMaterial ; 
class GSolid ; 
class GNode ;
class GBoundary ;
class GDrawable ; 

class GGeoOptiXGeometry  : public OptiXGeometry 
{
public:
    GGeoOptiXGeometry(GGeo* ggeo);
    virtual ~GGeoOptiXGeometry();

public:
    void convert();

private:
    void convertBoundaries();
    optix::Material convertBoundary(GBoundary* boundary);
    void addWavelengthTexture(optix::Material& material, GPropertyMap<float>* ptex);

private:
    void convertStructure();
    void traverseNode(GNode* node, unsigned int depth, bool recurse);

private:
    optix::Geometry convertGeometry(GSolid* solid);
    optix::GeometryInstance convertGeometryInstance(GSolid* solid);

public:
    optix::float3  getMin();
    optix::float3  getMax();
    optix::float3  getCenter();
    optix::float3  getExtent();
    optix::float3  getUp();

private:
    GGeo*        m_ggeo ; 

};








