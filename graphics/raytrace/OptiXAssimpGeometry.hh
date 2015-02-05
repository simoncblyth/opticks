#ifndef OPTIXASSIMPGEOMETRY_H
#define OPTIXASSIMPGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

class AssimpNode ; 

#include "AssimpWrap/AssimpGeometry.hh"

class OptiXAssimpGeometry  : public AssimpGeometry 
{
public:
    OptiXAssimpGeometry(const char* path);

    virtual ~OptiXAssimpGeometry();

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

private:

    optix::Material convertMaterial(aiMaterial* ai_material);

    optix::Geometry convertGeometry(aiMesh* mesh);

    void traverseNode(AssimpNode* node, unsigned int depth, bool recurse);

public:

    optix::float3  getMin();

    optix::float3  getMax();

    optix::float3  getCenter();

    optix::float3  getExtent();

    optix::float3  getUp();

    optix::Aabb getAabb();

private:

    optix::Context m_context ;

    optix::Material m_material ;

    optix::GeometryGroup m_geometry_group ; 

    std::vector<optix::Material> m_materials;

    std::vector<optix::Geometry> m_geometries;

    std::vector<optix::GeometryInstance> m_gis ;

};


#endif
