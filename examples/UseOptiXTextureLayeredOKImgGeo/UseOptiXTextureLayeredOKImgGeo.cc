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
TODO:

1. expt with realistic 2d theta-phi layered spherical texture : especially wrt the wrapping mode, passing domain range 
2. apply 2d theta-phi layered texture to some instanced geometry and shade based on it : eg make a variety of PPM beach-balls  

**/


const char* const CMAKE_TARGET =  "UseOptiXTextureLayeredOKImgGeo" ;


NPYBase* LoadPPMAsTextureArray(const char* path, const bool yflip  )
{
    unsigned ncomp_ = 4 ; 
    const char* config = "add_border" ; 
    NPY<unsigned char>* inp = ImageNPY::LoadPPM(path, yflip, ncomp_, config) ; 
    assert( inp->getDimensions() == 3 ); 
    LOG(info) << " original inp (height, width, ncomp)  " << inp->getShapeString() ; 

    unsigned layers = 1 ; 
    unsigned height = inp->getShape(0);  
    unsigned width = inp->getShape(1);  
    unsigned ncomp = inp->getShape(2);  

    inp->reshape(layers,height,width,ncomp) ; // unsure re height<->width layout 
    LOG(info) << " after reshape inp (layers,height,width,ncomp) " << inp->getShapeString()  ; 

    /* 
    inp->setMeta<float>("xmin", 0.f); 
    inp->setMeta<float>("xmax", 360.f); 
    inp->setMeta<float>("ymin", 0.f); 
    inp->setMeta<float>("ymax", 180.f); 
    */

    unsigned ncomp2 = inp->getShape(-1) ; 
    assert( ncomp2 == ncomp ); 

    return inp ; 
}


optix::Geometry CreateGeometry( optix::Context context )
{
    optix::Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );
    const char* ptx = OKConf::PTXPath( CMAKE_TARGET, "sphere.cu" ) ; 
    LOG(info) << "[ ptx " << ptx ; 
    optix::Program bd = context->createProgramFromPTXFile( ptx, "bounds" ) ;  
    optix::Program in = context->createProgramFromPTXFile( ptx, "intersect" ) ;  

    sphere->setBoundingBoxProgram(bd);
    sphere->setIntersectionProgram(in);
    sphere["sphere"]->setFloat( 0, 0, 0, 1.5 );
    LOG(info) << "] ptx " << ptx ; 
    return sphere;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* cu_name = SStr::Concat(CMAKE_TARGET, ".cu" ) ;  
    const char* main_ptx = OKConf::PTXPath( CMAKE_TARGET, cu_name ); 
    const char* tmpdir = SStr::Concat("$TMP/", CMAKE_TARGET ) ; 
    const char* path = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ;
    //const char* marchwood = "50.8919,-1.4483" ; 
    const char* null_island = "0.0,0.0" ; 
    const char* latlon_degrees = PLOG::instance->get_arg_after("--latlon", null_island ) ; 
    const char* tanyfov_ = PLOG::instance->get_arg_after("--tanyfov", "1.0" ) ; 

    glm::vec2 latlon_d = gvec2(latlon_degrees); 

    const float pi = glm::pi<float>() ;
    glm::vec2 latlon(latlon_d*pi/180.f); 
    const float tanYfov = gfloat_(tanyfov_) ; 
    LOG(info) << " latlon_d " << glm::to_string(latlon_d) << " tanYfov " << tanYfov ;  

    float dlon0 = -pi ;  // assume left edge of world texture is mid-pacific at -180 degrees longitude 

    float lat = latlon.x ;         // latitude, N:+ve, S:-ve,  zero at equator
    float lon = dlon0 + latlon.y ; // longitude, E:+ve W:-ve, zero at Greenwich 

    glm::vec3 eye_m( cosf(lat)*cosf(lon), cosf(lat)*sinf(lon),  sinf(lat) );   
    eye_m *= 1.1 ; 


    const bool yflip0 = false ; 
    NPYBase* inp = LoadPPMAsTextureArray(path, yflip0); 
    inp->save(tmpdir,"inp.npy"); 

    LOG(info) << " inp " << inp->getShapeString() ;
    unsigned height = inp->getShape(1);  
    unsigned width = inp->getShape(2);  

    LOG(info) 
        << " height " << height  
        << " width " << width
        ;  

    LOG(info) << " main_ptx: [" << main_ptx << "]" ; 

    // model frame : center-extent of model and viewpoint 
    glm::vec4 ce_m(      0.f,  0.f, 0.f, 1.5f );
    glm::vec3 look_m(    0.f,  0.f, 0.f );
    glm::vec3 up_m(      0.f,  0.f, 1.f );

    // world frame : eye point and view axes 
    glm::vec3 eye ;
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    const bool dump = true ; 

    nglmext::GetEyeUVW( ce_m, eye_m, look_m, up_m, width, height, tanYfov, eye, U, V, W, dump );

    std::cout << std::setw(10) << "ce_m"    << gpresent(ce_m) << std::endl ; 
    std::cout << std::setw(10) << "eye_m "  << gpresent(eye_m) << std::endl ; 
    std::cout << std::setw(10) << "look_m " << gpresent(look_m) << std::endl ; 
    std::cout << std::setw(10) << "up_m "   << gpresent(up_m) << std::endl ; 

    std::cout << std::setw(10) << "eye"  << gpresent(eye) << std::endl ; 
    std::cout << std::setw(10) << "U "   << gpresent(U) << std::endl ; 
    std::cout << std::setw(10) << "V "   << gpresent(V) << std::endl ; 
    std::cout << std::setw(10) << "W "   << gpresent(W) << std::endl ; 


    optix::Context context = optix::Context::create();
    context->setRayTypeCount(1); 
    context->setExceptionEnabled( RT_EXCEPTION_ALL , true );
    context->setPrintEnabled(1);  
    context->setPrintBufferSize(4096);
    context->setEntryPointCount(1);
    unsigned entry_point_index = 0u ; 

    const char* config = "INDEX_NORMALIZED_COORDINATES" ; 
    OTexture::Upload2DLayeredTexture<unsigned char>(context, "tex_param", "tex_domain", inp, config);   

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


    LOG(info) << "[ compile rg " ;  

    //const char* raygen = "raygen_reproduce_texture" ; 
    const char* raygen = "raygen" ; 
    optix::Program rg = context->createProgramFromPTXFile( main_ptx , raygen ) ; 
    LOG(info) << "] compile rg " ;  

    optix::Geometry g = CreateGeometry(context); 

    optix::Material m = context->createMaterial();

    LOG(info) << "[ compile ch " ;  
    optix::Program ch = context->createProgramFromPTXFile( main_ptx, "closest_hit_radiance0" ) ; 
    LOG(info) << "] compile ch " ;  

    m->setClosestHitProgram( entry_point_index, ch );

    optix::Program ex = context->createProgramFromPTXFile( main_ptx, "exception" ); 
    context->setExceptionProgram( entry_point_index,  ex );


    optix::GeometryInstance gi = context->createGeometryInstance( g, &m, &m+1 ) ;
    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1);
    gg->setChild( 0, gi );
    gg->setAcceleration( context->createAcceleration("Trbvh") );
    context["top_object"]->set( gg );


    context->setRayGenerationProgram( entry_point_index,  rg );  

    float near = 0.01f ;
    float scene_epsilon = near ;

    context[ "scene_epsilon"]->setFloat( scene_epsilon );
    context[ "eye"]->setFloat( eye.x, eye.y, eye.z  );
    context[ "U"  ]->setFloat( U.x, U.y, U.z  );
    context[ "V"  ]->setFloat( V.x, V.y, V.z  );
    context[ "W"  ]->setFloat( W.x, W.y, W.z  );
    context[ "radiance_ray_type"   ]->setUint( 0u );

    LOG(info) << "[ compile " ;  
    context->compile();  
    LOG(info) << "] compile " ;  

    LOG(info) << "[ validate " ;  
    context->validate();  
    LOG(info) << "] validate " ;  

    LOG(info) << "[ launch width height (" << width << " " << height << ") " ;  
    context->launch( entry_point_index , width, height  );
    LOG(info) << "] launch " ;  


    // Thinking about the way PPM images are stored in row-major order
    // makes me want to use (height, width, 4) 
    // BUT these are not images they are arrays 
     
    //unsigned s0 = width ; 
    //unsigned s1 = height ; 

    unsigned s0 = height ; 
    unsigned s1 = width ; 
    unsigned s2 = 4 ; 


    NPY<unsigned char>* out = NPY<unsigned char>::make(s0, s1, s2); 
    out->zero();
    void* out_data = outBuffer->map(); 
    out->read(out_data); 
    outBuffer->unmap(); 
    out->save(tmpdir,"out.npy"); 

    const bool yflip = true ; 
    ImageNPY::SavePPM(tmpdir, "out.ppm", out, yflip ); 

    NPY<float>* dbg = NPY<float>::make(s0, s1, s2); 
    dbg->zero();
    void* dbg_data = dbgBuffer->map(); 
    dbg->read(dbg_data); 
    dbgBuffer->unmap(); 
    dbg->save(tmpdir,"dbg.npy"); 

    NPY<float>* pos = NPY<float>::make(s0, s1, s2); 
    pos->zero();
    void* pos_data = posBuffer->map(); 
    pos->read(pos_data); 
    posBuffer->unmap(); 
    pos->save(tmpdir,"pos.npy"); 


    return 0 ; 
}


