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
UseOptiXGeometryTriangles
=============================

Minimally demonstrate OptiX geometry without using OXRAP.

* "standalone" ray traces a box using a normal shader


**/


#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include "OPTICKS_LOG.hh"
#include "OKConf.hh"
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NTrianglesNPY.hpp"
#include "SPPM.hh"




// Composition::getEyeUVW and examples/UseGeometryShader:getMVP
void getEyeUVW(const glm::vec4& ce, const unsigned width, const unsigned height, glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W )
{
    glm::vec3 tr(ce.x, ce.y, ce.z);  // ce is center-extent of model
    glm::vec3 sc(ce.w);
    glm::vec3 isc(1.f/ce.w);
    // model frame unit coordinates from/to world 
    glm::mat4 model2world = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);
    //glm::mat4 world2model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr);

   // View::getTransforms
    glm::vec4 eye_m( -1.f,-1.f,1.f,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.f, 0.f,0.f,1.f); 
    glm::vec4 up_m(   0.f, 0.f,1.f,1.f); 
    glm::vec4 gze_m( look_m - eye_m ) ; 

    const glm::mat4& m2w = model2world ; 
    glm::vec3 eye_ = glm::vec3( m2w * eye_m ) ; 
    //glm::vec3 look = glm::vec3( m2w * look_m ) ; 
    glm::vec3 up = glm::vec3( m2w * up_m ) ; 
    glm::vec3 gaze = glm::vec3( m2w * gze_m ) ;    

    glm::vec3 forward_ax = glm::normalize(gaze);
    glm::vec3 right_ax   = glm::normalize(glm::cross(forward_ax,up)); 
    glm::vec3 top_ax     = glm::normalize(glm::cross(right_ax,forward_ax));

    float aspect = float(width)/float(height) ;
    float tanYfov = 1.f ;  // reciprocal of camera zoom
    float gazelength = glm::length( gaze ) ;
    float v_half_height = gazelength * tanYfov ;
    float u_half_width  = v_half_height * aspect ;

    U = right_ax * u_half_width ;
    V = top_ax * v_half_height ;
    W = forward_ax * gazelength ; 
    eye = eye_ ; 
}


struct Solid
{
    NTrianglesNPY* trin ; 
    NPY<float>*     vtx ; 
    NPY<unsigned>*  idx ; 

    Solid() :
        trin(NTrianglesNPY::icosahedron()),
        vtx(NULL),
        idx(NULL)
    {
        NVtxIdx vtxidx ; 
        trin->to_vtxidx(vtxidx); 
        vtx = vtxidx.vtx ;  
        idx = vtxidx.idx ;  
    }
    void dump()
    {
        vtx->dump("vtx");
        idx->dump("idx");
    }
};


optix::GeometryGroup createGeometryTriangles(optix::Context& context, unsigned entry_point_index, const char* ptx)
{
    Solid solid ;  
    solid.dump(); 

    NPY<float>* vtx = solid.vtx ; 
    NPY<unsigned>* idx = solid.idx ;
 
    unsigned num_vertices = vtx->getShape(0); 
    unsigned num_faces = idx->getShape(0); 

    optix::Buffer vertex_buffer   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
    optix::Buffer index_buffer    = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, num_faces );

    memcpy( vertex_buffer->map(), vtx->getBytes(), vtx->getNumBytes(0) );
    memcpy( index_buffer->map(), idx->getBytes(), idx->getNumBytes(0) );

    vertex_buffer->unmap() ;
    index_buffer->unmap() ;

    optix::GeometryTriangles gtri = context->createGeometryTriangles();

    gtri->setPrimitiveCount( num_faces );
    gtri->setTriangleIndices( index_buffer, RT_FORMAT_UNSIGNED_INT3 );
    gtri->setVertices( num_vertices, vertex_buffer, RT_FORMAT_FLOAT3 );
    gtri->setBuildFlags( RTgeometrybuildflags( 0 ) );
    
    gtri["index_buffer"]->setBuffer( index_buffer );
    gtri["vertex_buffer"]->setBuffer( vertex_buffer );

    gtri->setAttributeProgram( context->createProgramFromPTXFile( ptx, "triangle_attributes" ) ); 

    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx, "closest_hit_radiance0" ));
    optix::GeometryInstance gi = context->createGeometryInstance( gtri, mat  ) ;  

    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1); 
    gg->setChild( 0, gi );
    gg->setAcceleration( context->createAcceleration("Trbvh") );

    return gg ; 
}

optix::GeometryGroup createGeometry(const glm::vec4& ce, optix::Context& context, unsigned entry_point_index, const char* ptx, const char* cmake_target)
{
    const char* box_ptx = OKConf::PTXPath( cmake_target, "box.cu" ) ; 
    optix::Geometry box = context->createGeometry();
    box->setPrimitiveCount( 1u );
    box->setBoundingBoxProgram( context->createProgramFromPTXFile( box_ptx , "box_bounds" ) );
    box->setIntersectionProgram( context->createProgramFromPTXFile( box_ptx , "box_intersect" ) ) ;

    float sz = ce.w ; 
    box["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
    box["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );

    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx, "closest_hit_radiance0" ));

    optix::GeometryInstance gi = context->createGeometryInstance( box, &mat, &mat+1 ) ;  

    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1); 
    gg->setChild( 0, gi );
    gg->setAcceleration( context->createAcceleration("Trbvh") );

    return gg ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    const char* cmake_target = "UseOptiXGeometryTriangles" ; 
    unsigned width = 1024u ; 
    unsigned height = 768 ; 

    glm::vec4 ce(0.,0.,0., 1.0); 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    getEyeUVW( ce, width, height, eye, U, V, W ); 


    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    context->setPrintEnabled(true); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 

    unsigned entry_point_index = 0u ;

    const char* ptx = OKConf::PTXPath( cmake_target, "UseOptiXGeometryTriangles.cu") ; 
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx , "miss" )); 

    //optix::GeometryGroup gg = createGeometry(ce, context, entry_point_index, ptx, cmake_target ) ; 
    optix::GeometryGroup gg = createGeometryTriangles( context, entry_point_index, ptx ) ; 

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

    const char* path = "/tmp/UseOptiXGeometryTriangles.ppm" ;  
    bool yflip = true ;  
    int ncomp = 4 ;   
    void* ptr = output_buffer->map() ; 
    SPPM::write(path,  (unsigned char*)ptr , width, height, ncomp, yflip );
    output_buffer->unmap(); 

    LOG(info) << argv[0] ; 
    return 0 ; 
}


