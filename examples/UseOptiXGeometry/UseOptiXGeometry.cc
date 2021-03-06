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
#include "OpticksCSG.h"
#include "OKConf.hh"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NPart.hpp"
#include "SPPM.hh"



void CreateInputUserBuffer(optix::Context& context, const char* key, NPYBase* array )
{
    // OGeo::CreateInputUserBuffer
    unsigned itemBytes = array->getNumBytes(1);  
    unsigned numBytes = array->getNumBytes() ;
    assert( numBytes % itemBytes == 0 );
    unsigned numItems = numBytes/itemBytes ;
    assert( array->getNumItems() == numItems ); 

    optix::Buffer buffer =  context->createBuffer( RT_BUFFER_INPUT ) ; 
    buffer->setFormat( RT_FORMAT_USER );
    buffer->setElementSize(itemBytes);
    buffer->setSize(numItems); 

    memcpy( buffer->map(), array->getBytes(), numBytes );
    buffer->unmap();

    context[key]->setBuffer(buffer); 
}


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
    else if( strcmp(geo_cu, "csg_intersect_primitive.cu") == 0 )
    {
        geo["sphere"]->setFloat( 0.f, 0.f, 0.f, sz ); 
    }
    else if( strcmp(geo_cu, "csg_intersect_part.cu") == 0 )
    {
        NPY<float>* parts = NPY<float>::make( 1, 4, 4);  
        parts->zero();   

        npart p0; 
        p0.zero(); 

        //p0.setTypeCode(CSG_SPHERE);
        //p0.setParam(0.f,0.f ,0.f, sz) ;

        /*
        {
            p0.setTypeCode(CSG_HYPERBOLOID);
            float r0 = sz/2.f ; 
            float zf = sz/2 ; 
            float z1 = -sz/2 ; 
            float z2 =  sz/2 ; 
            p0.setParam(r0,zf,z1,z2) ;
        }
        */
        {
            p0.setTypeCode(CSG_CONE);
            float r1 = 0.f ; 
            float z1 = -sz/2 ; 
            float r2 =  sz/2 ; 
            float z2 =  sz/2 ; 
            p0.setParam(r1,z1,r2,z2) ;
        }


        parts->setPart(p0, 0); 

        CreateInputUserBuffer(context, "partBuffer", parts ); 

        NPY<float>* trans = NPY<float>::make_identity_transforms(1); 
        NPY<float>* trips = NPY<float>::make_triple_transforms(trans); 
        trips->reshape(-1,4,4);   // GPU side expecting (n,4,4) 
        CreateInputUserBuffer(context, "tranBuffer", trips ); 

        NPY<float>* plans = NPY<float>::make_identity_transforms(1); 
        plans->reshape(-1,4);   // GPU side expecting (n,4) 
        CreateInputUserBuffer(context, "planBuffer", plans ); 
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


