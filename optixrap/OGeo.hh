/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <vector>

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "plog/Severity.h"

class RayTraceConfig ; 

class Opticks ; 

class OContext ; 

//class GGeo ; 
//class GGeoBase ; 
class GGeoLib ; 
class GMergedMesh ; 
class GBuffer ; 
template <typename S> class NPY ;

// used by OEngine::initGeometry

/**
OGeo
=====

Canonical OGeo instance resides in OScene and is
instanciated and has its *convert* called from OScene::init.
OScene::convert loops over the GMergedMesh within GGeo 
converting them into OptiX geometry groups. The first 
GMergedMesh is assumed to be non-instanced, the remainder
are expected to be instanced with appropriate 
transform and identity buffers.

Details of geometry tree are documented with the OGeo::convert method.

**/

#include "OGeoStat.hh"



struct OGeometry ; 

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API  OGeo 
{
public:


/*
    struct OGeometry 
    {
       optix::Geometry           g ; 
#if OPTIX_VERSION >= 60000
       optix::GeometryTriangles  gt ; 
#endif
       bool isGeometry() const ;  
       bool isGeometryTriangles() const ;  
    };
*/


    static const plog::Severity LEVEL ; 

    static const char* ACCEL ; 

    OGeo(OContext* ocontext, Opticks* ok, GGeoLib* geolib );
    void setTopGroup(optix::Group top);
    void setVerbose(bool verbose=true);
    std::string description() const ;
public:
    void convert();

private:
    void init();
    void convertMergedMesh(unsigned i);
    void dumpStats(const char* msg="OGeo::dumpStats");
public:
    template <typename T> static     optix::Buffer CreateInputUserBuffer(optix::Context& ctx, NPY<T>* src, unsigned elementSize, const char* name, const char* ctxname_informational, unsigned verbosity);
public:
    template <typename T>             optix::Buffer createInputBuffer(GBuffer* buf, RTformat format, unsigned int fold, const char* name, bool reuse=false);
    template <typename T, typename S> optix::Buffer createInputBuffer(NPY<S>*  buf, RTformat format, unsigned int fold, const char* name, bool reuse=false);
    
    template <typename T>            optix::Buffer createInputUserBuffer(NPY<T>* src, unsigned elementSize, const char* name);
private:
    optix::GeometryGroup   makeGlobalGeometryGroup(GMergedMesh* mm);
    optix::Group           makeRepeatedAssembly(GMergedMesh* mm );

private:
    void                     setTransformMatrix(optix::Transform& xform, const float* tdata ) ;
    optix::Acceleration      makeAcceleration(const char* accel, bool accel_props=false);
    optix::Material          makeMaterial();

    OGeometry*               makeOGeometry(GMergedMesh* mergedmesh);
    optix::GeometryInstance  makeGeometryInstance(OGeometry* geometry, optix::Material material, unsigned instance_index);
    optix::GeometryGroup     makeGeometryGroup(optix::GeometryInstance gi, optix::Acceleration accel );
private:
    optix::Geometry         makeAnalyticGeometry(GMergedMesh* mergedmesh);
    optix::Geometry         makeTriangulatedGeometry(GMergedMesh* mergedmesh);
#if OPTIX_VERSION >= 60000
    optix::GeometryTriangles  makeGeometryTriangles(GMergedMesh* mm);
#endif

private:
    void dump(const char* msg, const float* m);


private:
    // input references
    OContext*            m_ocontext ; 
    optix::Context       m_context ; 
    optix::Group         m_top ; 
    Opticks*             m_ok ; 
    int                  m_gltf ; 
    GGeoLib*             m_geolib ;  
    unsigned             m_verbosity ; 
private:
    // for giving "context names" to GPU buffer uploads
    const char*          getContextName() const ;
    unsigned             m_mmidx ; 
    const char*          m_top_accel ; 
    const char*          m_ggg_accel ; 
    const char*          m_assembly_accel ; 
    const char*          m_instance_accel ; 
private:
    // locals 
    RayTraceConfig*       m_cfg ; 
    std::vector<OGeoStat> m_stats ; 

};

