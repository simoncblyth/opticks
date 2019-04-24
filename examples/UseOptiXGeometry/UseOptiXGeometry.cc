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


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    const char* cmake_target = "UseOptiXGeometry" ; 
    unsigned width = 1024u ; 
    unsigned height = 768 ; 

    glm::vec4 ce(0.,0.,0., 0.5); 

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

    const char* ptx = OKConf::PTXPath( cmake_target, "UseOptiXGeometry.cu") ; 
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx , "miss" )); 
 
    const char* box_ptx = OKConf::PTXPath( cmake_target, "box.cu" ) ; 



    optix::Geometry box ; 
    assert( box.get() == NULL ); 

    box = context->createGeometry();
    assert( box.get() != NULL ); 


    box->setPrimitiveCount( 1u );
    box->setBoundingBoxProgram( context->createProgramFromPTXFile( box_ptx , "box_bounds" ) );
    box->setIntersectionProgram( context->createProgramFromPTXFile( box_ptx , "box_intersect" ) ) ;

    float sz = ce.w ; 
    box["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
    box["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );

    optix::Material box_mat = context->createMaterial();
    box_mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx, "closest_hit_radiance0" ));

    optix::GeometryInstance gi = context->createGeometryInstance( box, &box_mat, &box_mat+1 ) ;  
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


