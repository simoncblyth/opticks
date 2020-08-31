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

/**
UseOptiXGeometryOCtx
======================


**/

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include "OPTICKS_LOG.hh"
#include "OKConf.hh"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "ImageNPY.hpp"
#include "SStr.hh"
#include "SPPM.hh"

#define USE_OCTX 1

#ifdef USE_OCTX
#include "OCtx.hh"
#endif

const char* CMAKE_TARGET = "UseOptiXGeometryOCtx" ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    const char* geo_cu_default = "box.cu" ; 
    const char* geo_cu = argc > 1 ? argv[1] : geo_cu_default ;  

    const char* cu_name = SStr::Concat(CMAKE_TARGET, ".cu" ) ;
    const char* main_ptx = OKConf::PTXPath(CMAKE_TARGET, cu_name) ; 
    const char* geo_ptx = OKConf::PTXPath(CMAKE_TARGET, geo_cu ) ; 
    const char* tmpdir = SStr::Concat("$TMP/", CMAKE_TARGET ) ;

#ifdef USE_OCTX
    LOG(info) << " USE_OCTX enabled " ; 
    void* context_ptr = OCtx::Get()->ptr();  
    assert( context_ptr ); 
    //optix::Context context = optix::Context::take((RTcontext)context_ptr);
#else
    LOG(info) << " USE_OCTX NOT-enabled " ; 
    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    context->setPrintEnabled(true); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 
#endif

    unsigned entry_point_index = 0u ;

#ifdef USE_OCTX
    OCtx::Get()->set_raygen_program( entry_point_index, main_ptx, "raygen" ); 
    OCtx::Get()->set_miss_program( entry_point_index, main_ptx, "miss" ); 
#else
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( main_ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( main_ptx , "miss" )); 
#endif

    LOG(info) 
        << " CMAKE_TARGET " << CMAKE_TARGET
        << " geo_cu " << geo_cu 
        ; 

    unsigned width = 1024u ; 
    unsigned height = 768 ; 
    float tanYfov = 1.0f;
    bool dump = false ;  

    // model frame : center-extent of model and viewpoint 
    glm::vec4 ce_m(    0.f,  0.f, 0.f, 0.5f ); 
    glm::vec3 eye_m(  -1.f, -1.f, 1.f ); 
    glm::vec3 look_m(  0.f,  0.f, 0.f ); 
    glm::vec3 up_m(    0.f,  0.f, 1.f ); 
 
    // world frame : eye point and view axes 
    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    nglmext::GetEyeUVW( ce_m, eye_m, look_m, up_m, width, height, tanYfov, eye, U, V, W, dump ); 

    float sz = ce_m.w ; 

#ifdef USE_OCTX
    void* geo_ptr = OCtx::Get()->create_geometry( 1u, geo_ptx, "bounds", "intersect" ); 
    if( strcmp(geo_cu, "box.cu") == 0 )
    {
        OCtx::Get()->set_geometry_float3(geo_ptr, "boxmin", -sz/2.f, -sz/2.f, -sz/2.f  ); 
        OCtx::Get()->set_geometry_float3(geo_ptr, "boxmax",  sz/2.f,  sz/2.f,  sz/2.f  ); 
    }
    else if( strcmp(geo_cu, "sphere.cu") == 0 )
    {
        OCtx::Get()->set_geometry_float4(geo_ptr, "sphere", 0.f, 0.f, 0.f, sz ); 
    }
    //optix::Geometry geo = optix::Geometry::take((RTgeometry)geo_ptr); 
#else
    optix::Geometry geo ; 
    assert( geo.get() == NULL ); 

    geo = context->createGeometry();
    assert( geo.get() != NULL ); 

    geo->setPrimitiveCount( 1u );
    geo->setBoundingBoxProgram( context->createProgramFromPTXFile( geo_ptx , "bounds" ) );
    geo->setIntersectionProgram( context->createProgramFromPTXFile( geo_ptx , "intersect" ) ) ;

    if( strcmp(geo_cu, "box.cu") == 0 )
    {
        geo["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
        geo["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );
    }
    else if( strcmp(geo_cu, "sphere.cu") == 0 )
    {
        geo["sphere"]->setFloat( 0.f, 0.f, 0.f, sz ); 
    }
#endif

    const char* closest_hit = "closest_hit_radiance0" ; 
    const char* accel = "Trbvh" ; 
#ifdef USE_OCTX
    void* mat_ptr = OCtx::Get()->create_material(main_ptx, closest_hit, entry_point_index ); 
    void* gi_ptr = OCtx::Get()->create_geometryinstance( geo_ptr, mat_ptr ); 
    void* gg_ptr = OCtx::Get()->create_geometrygroup( gi_ptr ); 
    void* ac_ptr = OCtx::Get()->create_acceleration( accel ); 
    OCtx::Get()->set_geometrygroup_acceleration(gg_ptr, ac_ptr); 
    //optix::GeometryGroup gg = optix::GeometryGroup::take((RTgeometrygroup)gg_ptr); 
    OCtx::Get()->set_geometrygroup_context_variable("top_object", gg_ptr ); 
#else
    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( main_ptx, closest_hit ));
    optix::GeometryInstance gi = context->createGeometryInstance( geo, &mat, &mat+1 ) ;  
    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1); 
    gg->setChild( 0, gi );
    gg->setAcceleration( context->createAcceleration(accel) );
    context["top_object"]->set( gg );
#endif


    float scene_epsilon = 0.1f ; 
#ifdef USE_OCTX
    OCtx::Get()->set_context_viewpoint( eye, U, V, W, scene_epsilon );
#else
    context[ "scene_epsilon"]->setFloat( scene_epsilon ); 
    context[ "eye"]->setFloat( eye.x, eye.y, eye.z  );
    context[ "U"  ]->setFloat( U.x, U.y, U.z  );
    context[ "V"  ]->setFloat( V.x, V.y, V.z  );
    context[ "W"  ]->setFloat( W.x, W.y, W.z  );
    context[ "radiance_ray_type"   ]->setUint( 0u ); 
#endif


    NPY<unsigned char>* out = NPY<unsigned char>::make(height, width, 4);
#ifdef USE_OCTX
    OCtx::Get()->create_buffer(out, "output_buffer", 'O', ' ', -1);
#else
    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
    context["output_buffer"]->set( output_buffer );
#endif

#ifdef USE_OCTX
    OCtx::Get()->launch( entry_point_index , width, height  ); 
#else
    context->launch( entry_point_index , width, height  ); 
#endif

    out->zero(); 
#ifdef USE_OCTX
    OCtx::Get()->download_buffer(out, "output_buffer", -1);
#else
    void* out_data = output_buffer->map();
    out->read(out_data);
    output_buffer->unmap();
#endif

    const bool yflip = true ;
    ImageNPY::SavePPM(tmpdir, "out.ppm", out, yflip );
    LOG(info) << argv[0] <<  " writing to tmpdir  " << tmpdir  ; 
    return 0 ; 
}


