#ifndef GGEOOPTIXGEOMETRY_H
#define GGEOOPTIXGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "OptiXGeometry.hh"

class GGeo ; 
class GMaterial ; 
class GSolid ; 
class GNode ;
class GSubstance ;
class GPropertyMap ;
class GDrawable ; 

class GGeoOptiXGeometry  : public OptiXGeometry 
{
public:
    GGeoOptiXGeometry(GGeo* ggeo);
    virtual ~GGeoOptiXGeometry();

public:
    void convert();

private:
    void convertSubstances();
    optix::Material convertSubstance(GSubstance* substance);
    void addWavelengthTexture(optix::Material& material, GPropertyMap* ptex);

private:
    void convertStructure();
    void traverseNode(GNode* node, unsigned int depth, bool recurse);

private:
    optix::Geometry convertGeometry(GSolid* solid);
    optix::GeometryInstance convertGeometryInstance(GSolid* solid);
private:
    // start migration to GMergedMesh
    optix::Geometry convertDrawable(GDrawable* drawable);

public:
    optix::float3  getMin();
    optix::float3  getMax();
    optix::float3  getCenter();
    optix::float3  getExtent();
    optix::float3  getUp();

private:
    GGeo* m_ggeo ; 

};



#endif





