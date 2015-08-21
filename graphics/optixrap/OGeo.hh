#pragma once

#include <vector>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

//#include "OptiXGeometry.hh"


class GGeo ; 
class GMergedMesh ; 
class GBuffer ; 

// canonical usage from OptiXEngine::initGeometry
//
// TODO: rename to ?MeshGeometry? as handle multiple GMergedMesh 
//       with instancing support 
//

class OGeo 
{
public:
    OGeo(optix::Context& ctx, GGeo* gg);
    void setGeometryGroup(optix::GeometryGroup ggrp);
public:
    void convert();
    void setupAcceleration();

public:
    template <typename T>
    optix::Buffer createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold=1);

private:
    optix::GeometryInstance makeGeometryInstance(GMergedMesh* mergedmesh);
    optix::Geometry         makeGeometry(GMergedMesh* mergedmesh);

//public:
//    optix::float3  getMin();
//    optix::float3  getMax();

private:
    optix::Context       m_context ; 
    optix::GeometryGroup m_geometry_group ; 
    GGeo*                m_ggeo ; 
    std::vector<optix::GeometryInstance> m_gis ;

};


inline OGeo::OGeo(optix::Context& ctx, GGeo* gg)
           : 
           m_context(ctx),
           m_ggeo(gg)
{
}


