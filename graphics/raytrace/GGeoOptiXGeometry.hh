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

#include "GProperty.hh"

class GGeoOptiXGeometry  : public OptiXGeometry 
{
public:
    static const char* refractive_index ; 
    static const char* absorption_length ; 
    static const char* scattering_length ; 
    static const char* reemission_prob ; 

public:
    GGeoOptiXGeometry(GGeo* ggeo);
    virtual ~GGeoOptiXGeometry();

public:
    void convert();

private:
    void convertSubstances();
    optix::Material convertSubstance(GSubstance* substance);

private:
    void convertStructure();
    void traverseNode(GNode* node, unsigned int depth, bool recurse);

private:
    optix::Geometry convertGeometry(GSolid* solid);
    optix::GeometryInstance convertGeometryInstance(GSolid* solid);

private:
    void addWavelengthTexture(optix::Material& material, GSubstance* substance);
    void checkProperties(GPropertyMap* ptex);
    GPropertyD* getPropertyOrDefault(GPropertyMap* pmap, const char* pname);

private:
    GGeo* m_ggeo ; 

public:

    optix::float3  getMin();

    optix::float3  getMax();

    optix::float3  getCenter();

    optix::float3  getExtent();

    optix::float3  getUp();

};



#endif





