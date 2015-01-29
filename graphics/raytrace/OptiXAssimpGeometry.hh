#ifndef OPTIXASSIMPGEOMETRY_H
#define OPTIXASSIMPGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

class OptiXProgram ; 
class AssimpNode ; 

#include "AssimpGeometry.hh"

class OptiXAssimpGeometry  : public AssimpGeometry 
{
public:
    OptiXAssimpGeometry(const char* path);

    virtual ~OptiXAssimpGeometry();

public:

    void setContext(optix::Context& context);

    void setProgram(OptiXProgram* program);

    void convert(const char* query);

    void setupAcceleration();

    void setGeometryGroup(optix::GeometryGroup gg);

    optix::GeometryGroup getGeometryGroup();


private:

    optix::Material convertMaterial(aiMaterial* ai_material);

    optix::Geometry convertGeometry(aiMesh* mesh);

    void traverseNode(AssimpNode* node, unsigned int depth);

    unsigned int convertNode(AssimpNode* node);

    unsigned int convertSelection();

    void populateGeometryGroup();


private:

    OptiXProgram* m_program ; 

    optix::Context m_context ;

    optix::GeometryGroup m_geometry_group ; 

    std::vector<optix::Material> m_materials;

    std::vector<optix::Geometry> m_geometries;

    std::vector<optix::GeometryInstance> m_gis ;

public:

    optix::float3  getMin();

    optix::float3  getMax();

    optix::float3  getCenter();

    optix::float3  getExtent();

    optix::float3  getUp();

    optix::Aabb getAabb();

};


#endif
