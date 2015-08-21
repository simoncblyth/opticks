#pragma once

#include <vector>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>


class GGeo ; 
class GMergedMesh ; 
class GBuffer ; 

// used by OEngine::initGeometry

class OGeo 
{
public:
    static const char* BUILDER ; 
    static const char* TRAVERSER ; 

    OGeo(optix::Context& ctx, GGeo* gg);
    void setTop(optix::Group top);
public:
    void convert();
private:
    void init();

public:
    template <typename T> optix::Buffer createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold=1);
    optix::Group   makeRepeatedGroup(GMergedMesh* mm);

private:
    optix::Acceleration     makeAcceleration(const char* builder=NULL, const char* traverser=NULL);
    optix::GeometryInstance makeGeometryInstance(GMergedMesh* mergedmesh);
    optix::Geometry         makeGeometry(GMergedMesh* mergedmesh);
    void dump(const char* msg, const float* m);

private:
    // input references
    optix::Context       m_context ; 
    optix::Group         m_top ; 
    GGeo*                m_ggeo ; 
private:
    // locals 
    optix::GeometryGroup m_geometry_group ; 
    optix::Group         m_repeated_group ; 

};

inline OGeo::OGeo(optix::Context& ctx, GGeo* gg)
           : 
           m_context(ctx),
           m_ggeo(gg)
{
    init();
}


