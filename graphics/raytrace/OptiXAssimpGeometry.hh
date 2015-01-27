#ifndef OPTIXASSIMPGEOMETRY_H
#define OPTIXASSIMPGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

class OptiXProgram ; 

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

private:

    optix::Material convertMaterial(aiMaterial* ai_material);

    optix::Geometry convertGeometry(aiMesh* mesh);

    void traverseNode(aiNode* node);

    unsigned int convertNode(aiNode* node);

    unsigned int convertSelection();

    optix::GeometryGroup makeGeometryGroup();

private:

    OptiXProgram* m_program ; 

    optix::Context m_context ;

    std::vector<optix::Material> m_materials;

    std::vector<optix::Geometry> m_geometries;

    std::vector<optix::GeometryInstance> m_gis ;


};


#endif
