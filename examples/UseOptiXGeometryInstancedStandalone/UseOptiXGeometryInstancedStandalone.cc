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
UseOptiXGeometryInstancedStandalone
======================================


**/


#include <chrono>


#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>



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
    glm::vec4 eye_m(  0.f, 0.f, 0.1f,1.f);  //  viewpoint in unit model frame 
    glm::vec4 look_m( 0.7f, 0.7f, -0.7,1.f); 
    glm::vec4 up_m(   0.f, 0.f, 1.f,1.f); 
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


const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx" 
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

const char* PPMPath( const char* install_prefix, const char* stem, const char* ext=".ppm" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ppm/"
       << stem
       << ext
       ;
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

void SPPM_write( const char* filename, const unsigned char* image, int width, int height, int ncomp, bool yflip )
{
    FILE * fp; 
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int h=0; h < height ; h++ ) // flip vertically
    {   
        int y = yflip ? height - 1 - h : h ; 

        for( int x=0; x < width ; ++x ) 
        {
            *(data + (y*width+x)*3+0) = image[(h*width+x)*ncomp+0] ;   
            *(data + (y*width+x)*3+1) = image[(h*width+x)*ncomp+1] ;   
            *(data + (y*width+x)*3+2) = image[(h*width+x)*ncomp+2] ;   
        }
    }   
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    std::cout << "Wrote file (unsigned char*) " << filename << std::endl  ;
    delete[] data;
}


float angle_radians(float angle_degrees)
{
    return glm::pi<float>()*angle_degrees/180.f ; 
}

glm::mat4 make_transform(const std::string& order, const glm::vec3& tlat, const glm::vec4& axis_angle, const glm::vec3& scal )
{
    glm::mat4 mat(1.f) ;
    
    float angle = angle_radians(axis_angle.w) ; 

    for(unsigned i=0 ; i < order.length() ; i++)
    {   
        switch(order[i])
        {
           case 's': mat = glm::scale(mat, scal)         ; break ; 
           case 'r': mat = glm::rotate(mat, angle , glm::vec3(axis_angle)) ; break ; 
           case 't': mat = glm::translate(mat, tlat )    ; break ; 
        }
    }   
    // See tests/NGLMExtTests.cc:test_make_transform it shows that 
    // usually "trs" is the most convenient order to use
    // * what is confusing is that to get the translation done last, 
    //   needs to do glm::translate first 
    return mat  ;   
}

glm::mat4 make_transform(const std::string& order)
{
    glm::vec3 tla(0,0,100) ; 
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(1,1,1) ; 
    return make_transform(order, tla, rot, sca );
}


struct APIError
{   
    APIError( RTresult c, const std::string& f, int l ) 
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

// Error check/report helper for users of the C API 
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw APIError( code, __FILE__, __LINE__ );           \
  } while(0)


void InitRTX(int rtxmode)  
{
    if(rtxmode == -1)
    {
        //std::cerr << " rtx " << rtxmode << " leaving ASIS "  << std::endl  ;
    }
    else
    {
#if OPTIX_VERSION_MAJOR >= 6

        int rtx0(-1) ;
        RT_CHECK_ERROR( rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx0), &rtx0) );
        assert( rtx0 == 0 );  // despite being zero performance suggests it is enabled

        int rtx = rtxmode > 0 ? 1 : 0 ;
        //std::cerr << " rtx " << rtxmode << " setting  " << ( rtx == 1 ? "ON" : "OFF" ) << std::endl   ;
        RT_CHECK_ERROR( rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx), &rtx));

        int rtx2(-1) ;
        RT_CHECK_ERROR(rtGlobalGetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(rtx2), &rtx2));
        assert( rtx2 == rtx );
#else
        printf("RTX requires optix version >= 6 \n"); 
#endif

    }
}



int main(int argc, char** argv)
{
    const char* rtxstr = getenv("RTX");
    int rtxmode = rtxstr ? atoi(rtxstr) : -1 ; 
    assert( rtxmode == -1 || rtxmode == 0 || rtxmode == 1 ); 

    const char* ppmstr = getenv("PPM");
    int ppmsave = ppmstr ? atoi(ppmstr) : -1 ; 


    const char* name = getenv("STANDALONE_NAME") ; 
    assert( name && "expecting STANDALONE_NAME envvar with name of the CMake target " );
    const char* prefix = getenv("STANDALONE_PREFIX"); 
    assert( prefix && "expecting STANDALONE_PREFIX envvar pointing to writable directory" );

    const char* cmake_target = name ; 

    //unsigned factor = 2u ; 
    //unsigned width =  factor*2560u ; 
    //unsigned height = factor*1440u ; 

    //unsigned factor = 1u ; 
    //unsigned width =  factor*2880u ; 
    //unsigned height = factor*1800u ; 

    unsigned factor = 1u ; 
    unsigned width =  factor*1440u ; 
    unsigned height = factor*900u ; 


    const unsigned nu = 100u;
    const unsigned nv = 100u;
    const unsigned nw = 4u;

    float extent = 100.0 ; 
    glm::vec4 ce(float(nu),float(nv), 0.f, extent ); 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    getEyeUVW( ce, width, height, eye, U, V, W ); 

    InitRTX(rtxmode); 

    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    bool prnt = false ; 
    context->setPrintEnabled(prnt); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 

    unsigned entry_point_index = 0u ;
    const char* ptx = PTXPath( prefix, cmake_target, name ) ; 
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx , "miss" )); 


    float sz = 10.f ; 
    unsigned nbox = 10u ; 

#define RUBOX 1 

#ifdef RUBOX
    optix::Geometry rubox ; 
    rubox = context->createGeometry();
    rubox->setPrimitiveCount( nbox );
    const char* rubox_ptx = PTXPath( prefix, cmake_target, "rubox" ) ; 
    rubox->setBoundingBoxProgram( context->createProgramFromPTXFile( rubox_ptx , "rubox_bounds" ) );
    rubox->setIntersectionProgram( context->createProgramFromPTXFile( rubox_ptx , "rubox_intersect" ) ) ;
    optix::Geometry& instance = rubox ; 
#else
    optix::Geometry box ; 
    box = context->createGeometry();
    box->setPrimitiveCount( 1u );
    const char* box_ptx = PTXPath( prefix, cmake_target, "box" ) ; 
    box->setBoundingBoxProgram( context->createProgramFromPTXFile( box_ptx , "box_bounds" ) );
    box->setIntersectionProgram( context->createProgramFromPTXFile( box_ptx , "box_intersect" ) ) ;
    optix::Geometry& instance = box ; 
#endif

    instance["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
    instance["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );


    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx, "closest_hit_radiance0" ));

    optix::Group top = context->createGroup() ; 
    top->setAcceleration( context->createAcceleration( "Trbvh" ) ); 
    context["top_object"]->set( top );

    optix::Group assembly = context->createGroup();
    assembly->setChildCount( nu*nv*nw );
    assembly->setAcceleration( context->createAcceleration( "Trbvh" ) ); 
    top->addChild(assembly);

    optix::Acceleration instance_accel = context->createAcceleration( "Trbvh" );

    unsigned ichild(0); 
    for( unsigned u = 0; u < nu; ++u ) { 
    for( unsigned v = 0; v < nv ; ++v ) { 
    for( unsigned w = 0; w < nw ; ++w ) { 

        optix::Transform xform = context->createTransform();

        glm::vec4 rot( rand(), rand(), rand(),  rand()*360.f );
        //glm::vec4 rot(  0,  0, 1,  0 );
        glm::vec3 sca( 0.5 ) ; 
        glm::vec3 tla(  10.f*u , 10.f*v , -10.f*w ) ; 
        glm::mat4 m4 = make_transform("trs", tla, rot, sca );

        bool transpose = true ; 
        optix::Matrix4x4 m4_( glm::value_ptr(m4)  ) ;
        xform->setMatrix(transpose, m4_.getData(), 0); 

        assembly->setChild(ichild, xform);
        unsigned instance_index = ichild ;  
        ichild++ ;

        optix::GeometryInstance pergi = context->createGeometryInstance() ;
        pergi->setMaterialCount(1);
        pergi->setMaterial(0, mat );
        pergi->setGeometry( instance );
        //pergi["instance_index"]->setUint(instance_index);

        optix::GeometryGroup perxform = context->createGeometryGroup();
        perxform->addChild(pergi); 
        perxform->setAcceleration(instance_accel) ; 

        xform->setChild(perxform);
    }
    }
    }


    float near = 11.f ; 
    float scene_epsilon = near ; 

    context[ "scene_epsilon"]->setFloat( scene_epsilon ); 
    context[ "eye"]->setFloat( eye.x, eye.y, eye.z  );
    context[ "U"  ]->setFloat( U.x, U.y, U.z  );
    context[ "V"  ]->setFloat( V.x, V.y, V.z  );
    context[ "W"  ]->setFloat( W.x, W.y, W.z  );
    context[ "radiance_ray_type"   ]->setUint( 0u ); 

    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
    context["output_buffer"]->set( output_buffer );



    auto t0 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , 0, 0  ); 

    auto t1 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , width, height  ); 

    auto t2 = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> t_prelaunch = t1 - t0; 
    std::chrono::duration<double> t_launch    = t2 - t1; 

    std::cout 
         << " nbox " << nbox 
         << " rtxmode " << std::setw(2) << rtxmode 
         << " prelaunch " << std::setprecision(4) << std::fixed << std::setw(15) << t_prelaunch.count() 
         << " launch    " << std::setprecision(4) << std::fixed << std::setw(15) << t_launch.count() 
         << std::endl 
         ;


    if(ppmsave > 0)
    {
        const char* path = PPMPath( prefix, name ); 
        bool yflip = true ;  
        int ncomp = 4 ;   
        void* ptr = output_buffer->map() ; 
        SPPM_write(path,  (unsigned char*)ptr , width, height, ncomp, yflip );
        output_buffer->unmap(); 
    }

    return 0 ; 
}


