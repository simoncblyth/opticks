#pragma once

#include <vector>

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

class RayTraceConfig ; 

class Opticks ; 

class OContext ; 
class GGeo ; 
class GMergedMesh ; 
class GBuffer ; 
template <typename S> class NPY ;

// used by OEngine::initGeometry

/**
OGeo
=====

Crucial OptiX geometrical members:


*(optix::Group)m_top*

*(optix::GeometryGroup)m_geometry_group*

*(optix::Group)m_repeated_group*



**/


#include "OXRAP_API_EXPORT.hh"
class OXRAP_API  OGeo 
{
public:
    static const char* BUILDER ; 
    static const char* TRAVERSER ; 

    OGeo(OContext* ocontext, GGeo* gg, const char* builder=NULL, const char* traverser=NULL);
    void setTop(optix::Group top);
    void setVerbose(bool verbose=true);
    const char* description(const char* msg="OGeo::description");
public:
    void convert();
private:
    void init();

public:
    template <typename T>             optix::Buffer createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold, const char* name, bool reuse=false);
    template <typename T, typename S> optix::Buffer createInputBuffer(NPY<S>*  buf, RTformat format, unsigned int fold, const char* name, bool reuse=false);
    template<typename U>              optix::Buffer createInputUserBuffer(NPY<float>* src, const char* name);
public:
    optix::Group   makeRepeatedGroup(GMergedMesh* mm);
    //optix::Group   PRIOR_makeRepeatedGroup(GMergedMesh* mm, unsigned int limit=0);


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
    //optix::Buffer PRIOR_makeAnalyticGeometryIdentityBuffer(GMergedMesh* mm, unsigned int numSolidsMesh);
private:
    void dump(const char* msg, const float* m);


private:
    // input references
    OContext*            m_ocontext ; 
    optix::Context       m_context ; 
    optix::Group         m_top ; 
    GGeo*                m_ggeo ; 
    Opticks*             m_ok ; 
    const char*          m_builder ; 
    const char*          m_traverser ; 
    const char*          m_description ; 
private:
    // locals 
    optix::GeometryGroup m_geometry_group ; 
    optix::Group         m_repeated_group ; 
    RayTraceConfig*      m_cfg ; 
    bool                 m_verbose ; 

};

