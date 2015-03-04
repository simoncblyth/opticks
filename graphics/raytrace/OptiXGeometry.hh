#ifndef OPTIXGEOMETRY_H
#define OPTIXGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>


class OptiXGeometry   
{
public:
    OptiXGeometry();

    virtual ~OptiXGeometry();

public:

    void setContext(optix::Context& context);

    void setMaterial(optix::Material material);

    void setGeometryGroup(optix::GeometryGroup gg);

public:

    optix::Context getContext();

    optix::Material getMaterial();

    optix::GeometryGroup getGeometryGroup();

public:

    void addInstance(optix::Geometry geometry, optix::Material material);

    void setupAcceleration();

public:

    virtual optix::float3  getMin() = 0;

    virtual optix::float3  getMax() = 0;

public:

    optix::Aabb getAabb();

protected:

    optix::Context m_context ;

    optix::Material m_material ;

    optix::GeometryGroup m_geometry_group ; 

protected:

    std::vector<optix::Material> m_materials;

    std::vector<optix::Geometry> m_geometries;

    std::vector<optix::GeometryInstance> m_gis ;

};


#endif
