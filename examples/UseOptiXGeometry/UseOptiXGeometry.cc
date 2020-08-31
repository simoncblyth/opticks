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
UseOptiXGeometry
===================

Minimally demonstrate OptiX geometry without using OXRAP.

* "standalone" ray traces a box using a normal shader

**/

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include "OPTICKS_LOG.hh"
#include "OKConf.hh"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "SPPM.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    const char* geo_cu_default = "box.cu" ; 
    const char* geo_cu = argc > 1 ? argv[1] : geo_cu_default ;  
    LOG(info) << " geo_cu " << geo_cu ; 

    const char* cmake_target = "UseOptiXGeometry" ; 
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


    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    context->setPrintEnabled(true); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 

    unsigned entry_point_index = 0u ;

    const char* ptx = OKConf::PTXPath( cmake_target, "UseOptiXGeometry.cu") ; 
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx , "miss" )); 

    const char* geo_ptx = OKConf::PTXPath( cmake_target, geo_cu ) ; 

    optix::Geometry geo ; 
    assert( geo.get() == NULL ); 

    geo = context->createGeometry();
    assert( geo.get() != NULL ); 


    geo->setPrimitiveCount( 1u );
    geo->setBoundingBoxProgram( context->createProgramFromPTXFile( geo_ptx , "bounds" ) );
    geo->setIntersectionProgram( context->createProgramFromPTXFile( geo_ptx , "intersect" ) ) ;

    float sz = ce_m.w ; 

    if( strcmp(geo_cu, "box.cu") == 0 )
    {
        geo["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
        geo["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );
    }
    else if( strcmp(geo_cu, "sphere.cu") == 0 )
    {
        geo["sphere"]->setFloat( 0.f, 0.f, 0.f, sz ); 
    }



    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx, "closest_hit_radiance0" ));

    optix::GeometryInstance gi = context->createGeometryInstance( geo, &mat, &mat+1 ) ;  
    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1); 
    gg->setChild( 0, gi );
    gg->setAcceleration( context->createAcceleration("Trbvh") );

    context["top_object"]->set( gg );

    float near = 0.1f ; 
    float scene_epsilon = near ; 

    context[ "scene_epsilon"]->setFloat( scene_epsilon ); 
    context[ "eye"]->setFloat( eye.x, eye.y, eye.z  );
    context[ "U"  ]->setFloat( U.x, U.y, U.z  );
    context[ "V"  ]->setFloat( V.x, V.y, V.z  );
    context[ "W"  ]->setFloat( W.x, W.y, W.z  );
    context[ "radiance_ray_type"   ]->setUint( 0u ); 

    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
    context["output_buffer"]->set( output_buffer );

    context->launch( entry_point_index , width, height  ); 

    const char* path = "/tmp/UseOptiXGeometry.ppm" ;  
    bool yflip = true ;  
    int ncomp = 4 ;   
    void* ptr = output_buffer->map() ; 
    SPPM::write(path,  (unsigned char*)ptr , width, height, ncomp, yflip );
    output_buffer->unmap(); 

    LOG(info) << argv[0] ; 
    return 0 ; 
}


