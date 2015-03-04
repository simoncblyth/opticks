#ifndef ASSIMPOPTIXGEOMETRY_H
#define ASSIMPOPTIXGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include "OptiXGeometry.hh"

class AssimpNode ; 
class AssimpGeometry ; 
struct aiMaterial ; 
struct aiMesh ;

class AssimpOptiXGeometry  : public OptiXGeometry 
{
public:
    AssimpOptiXGeometry(AssimpGeometry* ageo);

    virtual ~AssimpOptiXGeometry();

public:

    void convert();

private:

    void convertMaterials();

    void convertStructure();

    optix::Material convertMaterial(aiMaterial* ai_material);

    optix::Geometry convertGeometry(aiMesh* mesh);

    void traverseNode(AssimpNode* node, unsigned int depth, bool recurse);

private:

    AssimpGeometry* m_ageo ; 

public:

    optix::float3  getMin();

    optix::float3  getMax();

    optix::float3  getCenter();

    optix::float3  getExtent();

    optix::float3  getUp();

};

#endif


