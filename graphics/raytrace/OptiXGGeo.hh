#ifndef OPTIXGGEO_H
#define OPTIXGGEO_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

class GGeo ; 

class OptiXGGeo 
{
public:
    OptiXGGeo(GGeo* gg);

    virtual ~OptiXGGeo();

public:

    void setContext(optix::Context& context);

    void setMaterial(optix::Material material);

    void setGeometryGroup(optix::GeometryGroup gg);

public:

    unsigned int getMaxDepth();

    optix::GeometryGroup getGeometryGroup();

    optix::Material getMaterial();

public:

    void convert();

    void setupAcceleration();

public:

    optix::float3  getMin();

    optix::float3  getMax();

    optix::float3  getCenter();

    optix::float3  getExtent();

    optix::float3  getUp();

    optix::Aabb getAabb();

private:

    GGeo* m_ggeo ;  

    optix::Context m_context ;

    optix::Material m_material ;

    optix::GeometryGroup m_geometry_group ; 

    std::vector<optix::Material> m_materials;

    std::vector<optix::Geometry> m_geometries;

    std::vector<optix::GeometryInstance> m_gis ;

};

#endif
