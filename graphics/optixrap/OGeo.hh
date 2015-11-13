#pragma once

#include <vector>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>


class RayTraceConfig ; 

class OContext ; 

class GGeo ; 
class GCache ; 
class GMergedMesh ; 
class GBuffer ; 
template <typename S> class NPY ;

// used by OEngine::initGeometry

class OGeo 
{
public:
    static const char* BUILDER ; 
    static const char* TRAVERSER ; 

    OGeo(OContext* ocontext, GGeo* gg, const char* builder=NULL, const char* traverser=NULL);
    void setTop(optix::Group top);
    const char* description(const char* msg="OGeo::description");
public:
    void convert();
private:
    void init();

public:
    template <typename T>             optix::Buffer createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold, const char* name, bool reuse=false);
    template <typename T, typename S> optix::Buffer createInputBuffer(NPY<S>*  buf, RTformat format, unsigned int fold, const char* name, bool reuse=false);
public:
    optix::Group   makeRepeatedGroup(GMergedMesh* mm);
    optix::Group   PRIOR_makeRepeatedGroup(GMergedMesh* mm, unsigned int limit=0);

private:
    optix::Acceleration     makeAcceleration(const char* builder=NULL, const char* traverser=NULL);
    optix::Geometry         makeGeometry(GMergedMesh* mergedmesh);
    optix::Material         makeMaterial();
    optix::GeometryInstance makeGeometryInstance(optix::Geometry geometry, optix::Material material);
    //optix::GeometryInstance makeGeometryInstance(GMergedMesh* mergedmesh);
private:
    optix::Geometry         makeAnalyticGeometry(GMergedMesh* mergedmesh);
    optix::Geometry         makeTriangulatedGeometry(GMergedMesh* mergedmesh);
private:
    optix::Buffer PRIOR_makeAnalyticGeometryIdentityBuffer(GMergedMesh* mm, unsigned int numSolidsMesh);
private:
    void dump(const char* msg, const float* m);


private:
    // input references
    OContext*            m_ocontext ; 
    optix::Context       m_context ; 
    optix::Group         m_top ; 
    GGeo*                m_ggeo ; 
    GCache*              m_cache ; 
    const char*          m_builder ; 
    const char*          m_traverser ; 
    const char*          m_description ; 
private:
    // locals 
    optix::GeometryGroup m_geometry_group ; 
    optix::Group         m_repeated_group ; 
    RayTraceConfig*      m_cfg ; 

};

inline OGeo::OGeo(OContext* ocontext, GGeo* gg, const char* builder, const char* traverser)
           : 
           m_ocontext(ocontext),
           m_ggeo(gg),
           m_cache(NULL),
           m_builder(builder ? strdup(builder) : BUILDER),
           m_traverser(traverser ? strdup(traverser) : TRAVERSER),
           m_description(NULL)
{
    init();
}

