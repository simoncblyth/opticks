/**
UseOptiXGeometryInstanced
==========================

Start from UseOptiXGeometryInstancedStandalone, plan:

1. DONE: adopt Opticks packages to reduce the amount of code
2. DONE: adopt OCtx watertight wrapper, adding whats needed for instancing  
3. DONE: add optional switch from box to sphere 
4. get a layered texture to work with instances, such that 
   different groups of instances use different layers of the texture 
5. DONE: generate PPM of thousands of textured Earths with some visible variation 
6. layered 1d float texture

Next onto UseOptiXGeometryInstancedOCtx starting with the OCtx branch of this, 
as its too difficult to do new things in two ways at once.

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
#else
// tex uploading : have no-alternative
#include "OCtx.hh"
#endif

const char* CMAKE_TARGET =  "UseOptiXGeometryInstanced" ;


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





NPY<float>* MakeTransforms(unsigned nu, unsigned nv, unsigned nw, const float scale )
{
    unsigned num_tr = nu*nv*nw ; 
    NPY<float>* tr = NPY<float>::make(num_tr, 4, 4); 
    tr->zero(); 
    bool transpose = false ; 
    unsigned count(0) ; 
    for( unsigned u = 0; u < nu; ++u ) { 
    for( unsigned v = 0; v < nv ; ++v ) { 
    for( unsigned w = 0; w < nw ; ++w ) { 

        glm::vec4 rot( rand(), rand(), rand(),  rand()*360.f );
        //glm::vec4 rot(  0,  0, 1,  0 );
        glm::vec3 sca( scale ) ; 
        glm::vec3 tla(  10.f*u , 10.f*v , -10.f*w ) ; 
        glm::mat4 m4 = nglmext::make_transform("trs", tla, rot, sca );

        tr->setMat4(m4, count, -1, transpose); 
        count++ ; 
    }
    }
    }
    assert( count == num_tr ); 
    return tr ; 
}


#ifdef USE_OCTX
#else
/**
MakeInstancedAssembly
-----------------------

      assembly (Group)
          xform_0                       (Transform)
             perxform_0                   (GeometryGroup)
                instance_accel                (Acceleration)
                pergi_0                       (GeometryInstance)
                    mat                           (Material)
                    instance                      (Geometry)

          xform_1                       (Transform)
             perxform_1                   (GeometryGroup)
                instance_accel                (Acceleration)
                pergi_1                       (GeometryInstance)
                    mat                           (Material)
                    instance                      (Geometry)

**/

optix::Group MakeInstancedAssembly(optix::Context context, NPY<float>* transforms, optix::Material mat, optix::Geometry instance)
{
    unsigned num_tr = transforms->getNumItems() ; 
    optix::Group assembly = context->createGroup();
    assembly->setChildCount( num_tr );
    assembly->setAcceleration( context->createAcceleration( "Trbvh" ) ); 

    optix::Acceleration instance_accel = context->createAcceleration( "Trbvh" );

    for(unsigned ichild=0 ; ichild < num_tr ; ichild++)
    { 
        glm::mat4 m4 = transforms->getMat4(ichild,-1); 

        bool transpose = true ; 
        optix::Transform xform = context->createTransform();
        xform->setMatrix(transpose, glm::value_ptr(m4), 0); 

        assembly->setChild(ichild, xform);
        ////unsigned instance_index = ichild ;  
        //ichild++ ;

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
    return assembly ; 
}

void LaunchInstrumented( optix::Context context, unsigned entry_point_index, unsigned width, unsigned height, double& t_prelaunch, double& t_launch  )
{
    auto t0 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , 0, 0  ); 

    auto t1 = std::chrono::high_resolution_clock::now();

    context->launch( entry_point_index , width, height  ); 

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_prelaunch_ = t1 - t0; 
    std::chrono::duration<double> t_launch_    = t2 - t1; 
    
    t_prelaunch = t_prelaunch_.count() ; 
    t_launch = t_launch_.count() ; 
}
#endif


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* rtxstr = getenv("RTX");
    int rtxmode = rtxstr ? atoi(rtxstr) : -1 ; 
    assert( rtxmode == -1 || rtxmode == 0 || rtxmode == 1 ); 

    const char* cu_name = SStr::Concat(CMAKE_TARGET, ".cu" ) ;   
    const char* main_ptx = OKConf::PTXPath( CMAKE_TARGET, cu_name );  
    const char* tmpdir = SStr::Concat("$TMP/", CMAKE_TARGET ) ; 

    bool with_tex(true); 
    NPY<unsigned char>* inp = NULL ; 
    if(with_tex)
    {
        const char* path = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ;
        const bool yflip0 = false ; 
        unsigned ncomp = 4 ; 
        const char* config0 = "add_border,add_midline,add_quadline" ; 
        bool layer_dimension = false ; 
        inp = ImageNPY::LoadPPM(path, yflip0, ncomp, config0, layer_dimension ) ; 
    }


    unsigned width =  512u ; 
    unsigned height = 256u ; 

    const unsigned nu = 100u;
    const unsigned nv = 100u;
    const unsigned nw = 4u;
    const float scale = 0.5f ; 

    NPY<float>* transforms = MakeTransforms(nu,nv,nw, scale); 
    unsigned num_transforms = transforms->getNumItems(); 
    assert( num_transforms == nu*nv*nw );  


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

    // CAUTION: when mixing use of OCtx and non-OCtx need to avoid creating two contexts, which can leading to black textures
    void* context_ptr = OCtx::Get()->ptr(); 
    optix::Context context = optix::Context::take((RTcontext)context_ptr);  

    // no non-OCtx alternative
    const char* tex_config = "INDEX_NORMALIZED_COORDINATES" ;
    unsigned tex_id = OCtx::Get()->upload_2d_texture("tex_param", inp, tex_config, -1); 
    LOG(info) << " tex_id " << tex_id ; 

#ifdef USE_OCTX
    LOG(info) << " USE_OCTX enabled " ; 
#else
    LOG(info) << " USE_OCTX NOT-enabled " ; 
#endif


    unsigned entry_point_index = 0u ;
#ifdef USE_OCTX
    OCtx::Get()->set_raygen_program( entry_point_index, main_ptx, "raygen" );
    OCtx::Get()->set_miss_program(   entry_point_index, main_ptx, "miss" );
#else
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( main_ptx , "raygen" )); 
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( main_ptx , "miss" )); 
#endif

    float sz = 10.f ; 
    unsigned nbox = 1u ; 

    const char* rubox_ptx = OKConf::PTXPath( CMAKE_TARGET, "rubox.cu" ) ; 
    const char* sphere_ptx = OKConf::PTXPath( CMAKE_TARGET, "sphere.cu" ) ; 

#ifdef USE_OCTX
    void* box_ptr = OCtx::Get()->create_geometry(nbox, rubox_ptx, "rubox_bounds", "rubox_intersect" ); 
    OCtx::Get()->set_geometry_float3(box_ptr, "boxmin", -sz/2.f, -sz/2.f, -sz/2.f ); 
    OCtx::Get()->set_geometry_float3(box_ptr, "boxmax",  sz/2.f,  sz/2.f,  sz/2.f ); 
    //optix::Geometry instance = optix::Geometry::take((RTgeometry)geo_ptr); 

    void* sph_ptr = OCtx::Get()->create_geometry(nbox, sphere_ptx, "bounds", "intersect" ); 
    //OCtx::Get()->set_geometry_float4( sph_ptr, "sphere",  0, 0, 0, 10.0 );
    OCtx::Get()->set_context_float4( "sphere",  0.f, 0.f, 0.f, 10.f );

    void* instance_ptr = sph_ptr ; 
    //void* instance_ptr = box_ptr ; 
#else
    optix::Geometry rubox ; 
    rubox = context->createGeometry();
    rubox->setPrimitiveCount( nbox );
    rubox->setBoundingBoxProgram( context->createProgramFromPTXFile( rubox_ptx , "rubox_bounds" ) );
    rubox->setIntersectionProgram( context->createProgramFromPTXFile( rubox_ptx , "rubox_intersect" ) ) ;
    rubox["boxmin"]->setFloat( -sz/2.f, -sz/2.f, -sz/2.f );
    rubox["boxmax"]->setFloat(  sz/2.f,  sz/2.f,  sz/2.f );

    optix::Geometry sphere ; 
    sphere = context->createGeometry();
    sphere->setPrimitiveCount(nbox);
    sphere->setBoundingBoxProgram( context->createProgramFromPTXFile( sphere_ptx , "bounds" ) );
    sphere->setIntersectionProgram( context->createProgramFromPTXFile( sphere_ptx , "intersect" ) );
    sphere["sphere"]->setFloat( 0, 0, 0, 10.0 );

    //optix::Geometry& instance = rubox ; 
    optix::Geometry& instance = sphere ; 
#endif

    //const char* closest_hit = "closest_hit_radiance0" ; 
    const char* closest_hit = "closest_hit_textured" ; 

#ifdef USE_OCTX
    void* mat_ptr = OCtx::Get()->create_material( main_ptx,  closest_hit, entry_point_index ); 
    void* assembly_ptr = OCtx::Get()->create_instanced_assembly( transforms, instance_ptr, mat_ptr );
    //optix::Material mat = optix::Material::take((RTmaterial)mat_ptr); 
    //optix::Group assembly = optix::Group::take((RTgroup)assembly_ptr); 
    void* top_ptr = OCtx::Get()->create_group("top_object", assembly_ptr );  
    void* top_accel = OCtx::Get()->create_acceleration("Trbvh");
    OCtx::Get()->set_group_acceleration( top_ptr, top_accel ); 

#else
    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( main_ptx, closest_hit ));
    optix::Group assembly = MakeInstancedAssembly(context, transforms, mat, instance); 
    optix::Group top = context->createGroup() ; 
    top->setAcceleration( context->createAcceleration( "Trbvh" ) ); 
    context["top_object"]->set( top );
    top->addChild(assembly);
#endif


    float near = 1.f ;   // when cut into earth, see circle of back to front geography from inside the sphere
    float scene_epsilon = near ; 
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
    void* outBuf = OCtx::Get()->create_buffer(out, "output_buffer", 'O', ' ', -1);
    OCtx::Get()->desc_buffer( outBuf );
#else
    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);
    context["output_buffer"]->set( output_buffer );
#endif

    LOG(info) << "[ launch " ; 
    double t_prelaunch ; 
    double t_launch ; 
#ifdef USE_OCTX
    OCtx::Get()->launch_instrumented( entry_point_index, width, height, t_prelaunch, t_launch );
#else
    LaunchInstrumented( context, entry_point_index, width, height, t_prelaunch, t_launch ); 
#endif
    LOG(info) << "] launch " ; 

    std::cout 
         << " nbox " << nbox 
         << " rtxmode " << std::setw(2) << rtxmode 
         << " prelaunch " << std::setprecision(4) << std::fixed << std::setw(15) << t_prelaunch 
         << " launch    " << std::setprecision(4) << std::fixed << std::setw(15) << t_launch 
         << std::endl 
         ;

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

    return 0 ; 
}


