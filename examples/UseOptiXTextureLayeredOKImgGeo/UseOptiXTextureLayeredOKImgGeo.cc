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

#include <array>
#include <sstream>
#include <string>
#include <cstring>
#include <cassert>

#define USE_OCTX 1 

#ifdef USE_OCTX
#include "OTex.hh"
#include "OCtx.hh"
#endif


#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include <iostream>
#include "OKConf.hh"
#include "SStr.hh"
#include "OFormat.hh"
#include "OTexture.hh"
#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "ImageNPY.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

/**
UseOptiXTextureLayeredOKImgGeo
================================

This applies a 2d theta-phi texture to a sphere.  When using a texture map of Earth this
creates orthographic views centered on any latitude, longitude.::

   UseOptiXTextureLayeredOKImgGeo ~/opticks_refs/Earth_Albedo_8192_4096.ppm --latlon shijingshan --tanyfov 0.4
   open /tmp/blyth/opticks/UseOptiXTextureLayeredOKImgGeo/out.ppm
   UseOptiXTextureLayeredOKImgGeo ~/opticks_refs/Earth_Albedo_8192_4096.ppm --latlon 12.5779,122.2691 --tanyfov 0.4
   open /tmp/blyth/opticks/UseOptiXTextureLayeredOKImgGeo/out.ppm

**/

const char* const CMAKE_TARGET =  "UseOptiXTextureLayeredOKImgGeo" ;

/**
get_latitude_longitude_radians
-------------------------------

Converts "latitude,longitude" strings in degrees or shortcut placenames 
into (latitude,longitude) in radians 

**/

glm::vec2 get_latitude_longitude_radians(const char* q_ )
{
    std::map<std::string, std::string> name2latlon ;   
    name2latlon["marchwood"] = "50.8919,-1.4483" ; 
    name2latlon["nullisland"] = "0.0,0.0" ; 
    name2latlon["shijingshan"] = "39.9066,116.2230" ; 
    name2latlon["romblon"] = "12.5779,122.2691" ; 

    std::string q = q_ ; 
    const char* latlon_degrees = name2latlon.find(q) != name2latlon.end() ? name2latlon[q].c_str() : q.c_str() ; 
    glm::vec2 latlon_d = gvec2(latlon_degrees); 
    const float pi = glm::pi<float>() ;

    glm::vec2 latlon(latlon_d*pi/180.f); 

    LOG(info) 
        << " q_ " << q_
        << " latlon_d " << glm::to_string(latlon_d) 
        << " latlon "   << glm::to_string(latlon) 
        ;  

    return latlon ; 
}

#ifdef USE_OCTX
#else
optix::Geometry CreateGeometry( optix::Context context, unsigned prim_count, const char* ptxpath, const char* bounds_func, const char* intersect_func )
{
    optix::Geometry geom = context->createGeometry();
    geom->setPrimitiveCount( prim_count );
    LOG(info) << "[ ptxpath " << ptxpath ; 
    optix::Program bd = context->createProgramFromPTXFile( ptxpath, bounds_func ) ;  
    optix::Program in = context->createProgramFromPTXFile( ptxpath, intersect_func ) ;  
    geom->setBoundingBoxProgram(bd);
    geom->setIntersectionProgram(in);
    LOG(info) << "] ptxpath " << ptxpath ; 
    return geom;
}

optix::Material CreateMaterial( optix::Context context, const char* ptxpath, const char* closest_hit_func, unsigned entry_point_index )
{
    optix::Material mat = context->createMaterial();
    LOG(info) << "[ compile ch " ;  
    optix::Program ch = context->createProgramFromPTXFile( ptxpath, closest_hit_func ) ; 
    LOG(info) << "] compile ch " ;  
    mat->setClosestHitProgram( entry_point_index, ch );
    return mat ; 
}
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cu_name = SStr::Concat(CMAKE_TARGET, ".cu" ) ;  
    const char* main_ptx = OKConf::PTXPath( CMAKE_TARGET, cu_name ); 
    const char* tmpdir = SStr::Concat("$TMP/", CMAKE_TARGET ) ; 
    const char* path = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ;

    const char* tanyfov_ = PLOG::instance->get_arg_after("--tanyfov", "1.0" ) ; 
    const float tanYfov = gfloat_(tanyfov_) ; 

    const char* latlon_q = PLOG::instance->get_arg_after("--latlon", "0,0" ) ; 
    glm::vec2 latlon(get_latitude_longitude_radians(latlon_q)); 

    const float pi = glm::pi<float>() ;
    float dlon0 = -pi ;            // assume left edge of world texture is mid-pacific at -180 degrees longitude 
    float lat = latlon.x ;         // latitude, N:+ve, S:-ve, zero at equator
    float lon = dlon0 + latlon.y ; // longitude, E:+ve W:-ve, zero at Greenwich prime meridian

    glm::vec3 eye_m( cosf(lat)*cosf(lon), cosf(lat)*sinf(lon),  sinf(lat) );   
    eye_m *= 1.1 ; 

    const bool yflip0 = false ; 
    unsigned ncomp0 = 4 ; 
    const char* config0 = "add_border" ; 
    bool layer_dimension = false ; 
    NPY<unsigned char>* inp = ImageNPY::LoadPPM(path, yflip0, ncomp0, config0, layer_dimension ) ; 
    inp->save(tmpdir,"inp.npy"); 

    int height = inp->getShape(0);  
    int width = inp->getShape(1);  
    int ncomp = inp->getShape(2); 

    LOG(info) << " inp " << inp->getShapeString()
              << " height " << height  
              << " width " << width 
              << " ncomp " << ncomp
              ;  

    std::vector<int> shape = { height, width, 4 };

    glm::vec4 ce_m(      0.f,  0.f, 0.f, 1.5f ); // model frame : center-extent of model and viewpoint 
    glm::vec3 look_m(    0.f,  0.f, 0.f );
    glm::vec3 up_m(      0.f,  0.f, 1.f );

    glm::vec3 eye ; // world frame : eye point and view axes 
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    const bool dump = true ; 
    nglmext::GetEyeUVW( ce_m, eye_m, look_m, up_m, width, height, tanYfov, eye, U, V, W, dump );

    const char* tex_config = "INDEX_NORMALIZED_COORDINATES" ; 
#ifdef USE_OCTX
    OCtx::Get();  
    OCtx::Get()->upload_2d_texture("tex_param", inp, tex_config, -1);   
    //optix::Context context = optix::Context::take((RTcontext)OCtx_get()) ;  // interim kludge until everything is wrapped 
#else
    optix::Context context = optix::Context::create();
    context->setRayTypeCount(1); 
    context->setExceptionEnabled( RT_EXCEPTION_ALL , true );
    context->setPrintEnabled(1);  
    context->setPrintBufferSize(4096);
    context->setEntryPointCount(1);
    OTexture::Upload2DLayeredTexture<unsigned char>(context, "tex_param", "tex_domain", inp, tex_config);   
#endif
    unsigned entry_point_index = 0u ; 


    NPY<unsigned char>* out = NPY<unsigned char>::make(shape); 
    NPY<float>* dbg = NPY<float>::make(shape); 
    NPY<float>* pos = NPY<float>::make(shape); 
    
#ifdef USE_OCTX
    void* outBuf = OCtx::Get()->create_buffer(out, "out_buffer", 'O', ' ', -1); 
    OCtx::Get()->desc_buffer( outBuf );  

    void* dbgBuf = OCtx::Get()->create_buffer(dbg, "dbg_buffer", 'O', ' ', -1); 
    OCtx::Get()->desc_buffer( dbgBuf );  

    void* posBuf = OCtx::Get()->create_buffer(pos, "pos_buffer", 'O', ' ', -1); 
    OCtx::Get()->desc_buffer( posBuf );  
#else
    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT); 
    outBuffer->setFormat( RT_FORMAT_UNSIGNED_BYTE4); 
    outBuffer->setSize(width, height); 
    context["out_buffer"]->setBuffer(outBuffer); 

    optix::Buffer dbgBuffer = context->createBuffer(RT_BUFFER_OUTPUT); 
    dbgBuffer->setFormat( RT_FORMAT_FLOAT4 ); 
    dbgBuffer->setSize(width, height); 
    context["dbg_buffer"]->setBuffer(dbgBuffer); 

    optix::Buffer posBuffer = context->createBuffer(RT_BUFFER_OUTPUT); 
    posBuffer->setFormat( RT_FORMAT_FLOAT4 ); 
    posBuffer->setSize(width, height); 
    context["pos_buffer"]->setBuffer(posBuffer); 
#endif

    //const char* raygen = "raygen_reproduce_texture" ; 
    const char* raygen = "raygen" ; 
    LOG(info) << "[ compile rg + ex " ;  
#ifdef USE_OCTX
    OCtx::Get()->set_raygen_program( entry_point_index, main_ptx, raygen ); 
    OCtx::Get()->set_exception_program( entry_point_index, main_ptx, "exception" ); 
#else
    optix::Program rg = context->createProgramFromPTXFile( main_ptx , raygen ) ; 
    context->setRayGenerationProgram( entry_point_index,  rg );  

    optix::Program ex = context->createProgramFromPTXFile( main_ptx, "exception" ); 
    context->setExceptionProgram( entry_point_index,  ex );
#endif
    LOG(info) << "] compile rg + ex " ;  

    float scene_epsilon = 0.01f ;
#ifdef USE_OCTX
    OCtx::Get()->set_context_viewpoint( eye, U, V, W, scene_epsilon ); 
#else
    rg[ "scene_epsilon"]->setFloat( scene_epsilon );
    rg[ "eye"]->setFloat( eye.x, eye.y, eye.z  );
    rg[ "U"  ]->setFloat( U.x, U.y, U.z  );
    rg[ "V"  ]->setFloat( V.x, V.y, V.z  );
    rg[ "W"  ]->setFloat( W.x, W.y, W.z  );
    rg[ "radiance_ray_type"   ]->setUint( 0u );
#endif

    const char* sphere_ptx = OKConf::PTXPath( CMAKE_TARGET, "sphere.cu" ) ; 
    const char* ptxpath = sphere_ptx ; 
#ifdef USE_OCTX
    void* geo_ptr = OCtx::Get()->create_geometry(1u, ptxpath, "bounds", "intersect" ); 
    OCtx::Get()->set_geometry_float4( geo_ptr, "sphere",  0, 0, 0, 1.5 );   

    void* mat_ptr = OCtx::Get()->create_material( main_ptx, "closest_hit_radiance0", entry_point_index );
    void* gi_ptr  = OCtx::Get()->create_geometryinstance(geo_ptr, mat_ptr);  

    //optix::Geometry geo = optix::Geometry::take((RTgeometry)geo_ptr); 
    //optix::Material mat = optix::Material::take((RTmaterial)mat_ptr);  
    //optix::GeometryInstance gi = optix::GeometryInstance::take((RTgeometryinstance)gi_ptr);  
    //optix::GeometryGroup gg = optix::GeometryGroup::take((RTgeometrygroup)gg_ptr);  

    std::vector<const void*> vgi = { gi_ptr } ; 
    void* gg_ptr = OCtx::Get()->create_geometrygroup( vgi ); 
    void* ac_ptr = OCtx::Get()->create_acceleration( "Trbvh" ); 
    OCtx::Get()->set_geometrygroup_acceleration( gg_ptr, ac_ptr ); 
    OCtx::Get()->set_geometrygroup_context_variable("top_object", gg_ptr ); 

    OCtx::Get()->compile();  
    OCtx::Get()->validate();  
    OCtx::Get()->launch(entry_point_index , width, height); 
#else
    optix::Geometry geo = CreateGeometry(context, 1u, ptxpath, "bounds", "intersect" ); 
    geo["sphere"]->setFloat( 0, 0, 0, 1.5 );
    optix::Material mat = CreateMaterial(context, main_ptx, "closest_hit_radiance0", entry_point_index ); 
    optix::GeometryInstance gi = context->createGeometryInstance( geo, &mat, &mat+1 ) ;

    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1);
    gg->setChild( 0, gi );

    gg->setAcceleration( context->createAcceleration("Trbvh") );
    context["top_object"]->set( gg );

    LOG(info) << "[ compile " ;  
    context->compile();  
    LOG(info) << "] compile " ;  

    LOG(info) << "[ validate " ;  
    context->validate();  
    LOG(info) << "] validate " ;  

    LOG(info) << "[ launch width height (" << width << " " << height << ") " ;  
    context->launch( entry_point_index , width, height  );
    LOG(info) << "] launch " ;  
#endif


    out->zero();
    dbg->zero();
    pos->zero();

#ifdef USE_OCTX
    OCtx::Get()->download_buffer(out, "out_buffer", -1);
    OCtx::Get()->download_buffer(dbg, "dbg_buffer", -1);
    OCtx::Get()->download_buffer(pos, "pos_buffer", -1);
#else
    void* out_data = outBuffer->map(); 
    out->read(out_data); 
    outBuffer->unmap(); 

    void* dbg_data = dbgBuffer->map(); 
    dbg->read(dbg_data); 
    dbgBuffer->unmap(); 

    void* pos_data = posBuffer->map(); 
    pos->read(pos_data); 
    posBuffer->unmap(); 
#endif
    out->save(tmpdir,"out.npy"); 
    dbg->save(tmpdir,"dbg.npy"); 
    pos->save(tmpdir,"pos.npy"); 
    const bool yflip = true ; 
    ImageNPY::SavePPM(tmpdir, "out.ppm", out, yflip ); 
    return 0 ; 
}

