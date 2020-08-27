/**
UseOptiXGeometryInstanced
==========================

Start from UseOptiXGeometryInstancedStandalone, plan:

1. adopt Opticks packages to reduce the amount of code
2. adopt OCtx watertight wrapper, adding whats needed for instancing  
3. add optional switch from box to sphere 
4. get a layered texture to work with instances, such that 
   different groups of instances use different layers of the texture 
5. generate PPM of thousands of textured Earths with some visible variation 

**/

#include <chrono>

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <fstream>

#include "OKConf.hh"
#include "SStr.hh"
#include "OPTICKS_LOG.hh"
#include "OXPPNS.hh"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "ImageNPY.hpp"

#define USE_OCTX 1 

#ifdef USE_OCTX
#include "OCtx.hh"
#endif



const char* CMAKE_TARGET =  "UseOptiXGeometryInstanced" ;

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
    OPTICKS_LOG(argc, argv); 

    const char* rtxstr = getenv("RTX");
    int rtxmode = rtxstr ? atoi(rtxstr) : -1 ; 
    assert( rtxmode == -1 || rtxmode == 0 || rtxmode == 1 ); 

    const char* cu_name = SStr::Concat(CMAKE_TARGET, ".cu" ) ;   
    const char* main_ptx = OKConf::PTXPath( CMAKE_TARGET, cu_name );  
    const char* tmpdir = SStr::Concat("$TMP/", CMAKE_TARGET ) ; 

    unsigned factor = 1u ; 
    unsigned width =  factor*1440u ; 
    unsigned height = factor*900u ; 

    const unsigned nu = 100u;
    const unsigned nv = 100u;
    const unsigned nw = 4u;

    float extent = 100.0 ; 
    glm::vec4 ce_m(float(nu),float(nv), 0.f, extent ); 
    glm::vec3 eye_m(  0.f, 0.f, 0.1f  );  //  viewpoint in unit model frame 
    glm::vec3 look_m( 0.7f, 0.7f, -0.7); 
    glm::vec3 up_m(   0.f, 0.f, 1.f   ); 

    float tanYfov = 1.f ; 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 

    const bool dump = true ;
    nglmext::GetEyeUVW( ce_m, eye_m, look_m, up_m, width, height, tanYfov, eye, U, V, W, dump );


    InitRTX(rtxmode); 


#ifdef USE_OCTX
    optix::Context context = optix::Context::take((RTcontext)OCtx_get()) ;  // interim kludge until everything is wrapped 
#else
    optix::Context context = optix::Context::create();

    context->setRayTypeCount(1); 

    bool prnt = false ; 
    context->setPrintEnabled(prnt); 
    //context->setPrintLaunchIndex(5,0,0); 
    context->setPrintBufferSize(4096); 
    context->setEntryPointCount(1); 
#endif


    unsigned entry_point_index = 0u ;
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( main_ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( main_ptx , "miss" )); 

    float sz = 10.f ; 
    unsigned nbox = 10u ; 

#define RUBOX 1 

#ifdef RUBOX
    optix::Geometry rubox ; 
    rubox = context->createGeometry();
    rubox->setPrimitiveCount( nbox );
    const char* rubox_ptx = OKConf::PTXPath( CMAKE_TARGET, "rubox.cu" ) ; 
    rubox->setBoundingBoxProgram( context->createProgramFromPTXFile( rubox_ptx , "rubox_bounds" ) );
    rubox->setIntersectionProgram( context->createProgramFromPTXFile( rubox_ptx , "rubox_intersect" ) ) ;
    optix::Geometry& instance = rubox ; 
#else
    optix::Geometry box ; 
    box = context->createGeometry();
    box->setPrimitiveCount( 1u );
    const char* box_ptx = OKConf::PTXPath( CMAKE_TARGET, "box.cu" ) ; 
    box->setBoundingBoxProgram( context->createProgramFromPTXFile( box_ptx , "box_bounds" ) );
    box->setIntersectionProgram( context->createProgramFromPTXFile( box_ptx , "box_intersect" ) ) ;
    optix::Geometry& instance = box ; 
#endif

    instance["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
    instance["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );


    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( main_ptx, "closest_hit_radiance0" ));

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
        //unsigned instance_index = ichild ;  
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


    NPY<unsigned char>* out = NPY<unsigned char>::make(height, width, 4);
#ifdef USE_OCTX
    void* outBuf = OCtx_create_buffer(out, "output_buffer", 'O', ' ');
    OCtx_desc_buffer( outBuf );
#else
    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
    context["output_buffer"]->set( output_buffer );
#endif



    LOG(info) << "[ launch " ; 

    auto t0 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , 0, 0  ); 

    auto t1 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , width, height  ); 

    auto t2 = std::chrono::high_resolution_clock::now();

    LOG(info) << "] launch " ; 


    std::chrono::duration<double> t_prelaunch = t1 - t0; 
    std::chrono::duration<double> t_launch    = t2 - t1; 

    std::cout 
         << " nbox " << nbox 
         << " rtxmode " << std::setw(2) << rtxmode 
         << " prelaunch " << std::setprecision(4) << std::fixed << std::setw(15) << t_prelaunch.count() 
         << " launch    " << std::setprecision(4) << std::fixed << std::setw(15) << t_launch.count() 
         << std::endl 
         ;


#ifdef USE_OCTX
    OCtx_download_buffer(out, "output_buffer");
#else
    out->zero();
    void* out_data = output_buffer->map();
    out->read(out_data);
    output_buffer->unmap();
#endif

    const bool yflip = true ;
    ImageNPY::SavePPM(tmpdir, "out.ppm", out, yflip );

    return 0 ; 
}


