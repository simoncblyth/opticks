#ifndef GGEOOPTIXGEOMETRY_H
#define GGEOOPTIXGEOMETRY_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include "OptiXGeometry.hh"

class GGeo ; 
class GMaterial ; 

class GGeoOptiXGeometry  : public OptiXGeometry 
{
public:
    GGeoOptiXGeometry(GGeo* ggeo);

    virtual ~GGeoOptiXGeometry();

public:

    void convert();

private:

    void convertMaterials();

    void convertStructure();

    optix::Material convertMaterial(GMaterial* gmat);

    //optix::Geometry convertGeometry(aiMesh* mesh);

    //void traverseNode(AssimpNode* node, unsigned int depth, bool recurse);

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





