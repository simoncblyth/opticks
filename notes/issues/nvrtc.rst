nvrtc
=======

Resource specialization of kernels would benefit from runtime compilation 
as can then plant tokens into the source which are resolved based on a replacement map.  
This enables static constants in the source to become effectively dynamic. 
Could find and replace tokens manually or using string templating engine such as 

* https://github.com/pantor/inja


* could do this compilation just at geometry translation time, not at every launch, if it takes too long  
  

what about headers that include headers
------------------------------------------

* https://stackoverflow.com/questions/40087364/how-do-you-include-standard-cuda-libraries-to-link-with-nvrtc-code

Hmm it seems its not so easy to use curand with nvrtc
-------------------------------------------------------

* https://stackoverflow.com/questions/40087364/how-do-you-include-standard-cuda-libraries-to-link-with-nvrtc-code

* not necessarily a show stopper, it is OptiX geometry code that needs NVRTC not the curand using simulation node

* https://forums.developer.nvidia.com/t/using-curand-inside-nvrtc-jit-compiled-kernels/193826


optix nvrtc : maybe forces use of more than 700
------------------------------------------------------

Yep. Release notes:: 

    7.1.0 :Fixed support in optix headers for cuda runtime compilation using nvrtc.


* https://forums.developer.nvidia.com/t/optix-7-samples-using-nvrtc/79800



jitify
--------

* https://github.com/NVIDIA/jitify



UseNVRTC
---------------

* 2022 Mar : pulled out Prog as starting point at trying to use non-trivially 


Looked at this before in 

* opticks/examples/UseNVRTC



::

    epsilon:SDK blyth$ optix7-fl nvrtc
    ./CMakeLists.txt
    ./CMake/FindCUDA.cmake
    ./sampleConfig.h.in
    ./sutil/CMakeLists.txt
    ./sutil/sutil.cpp
    ./sutil/sutil.h
    epsilon:SDK blyth$ 


sutil/sutil.cpp::

    static void getCuStringFromFile( std::string& cu, std::string& location, const char* sample_name, const char* filename )
    static void getPtxFromCuString( std::string& ptx, const char* sample_name, const char* cu_source, const char* name, const char** log_string )


::

    747 struct PtxSourceCache
    748 {
    749     std::map<std::string, std::string*> map;
    750     ~PtxSourceCache()
    751     {
    752         for( std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it )
    753             delete it->second;
    754     }
    755 };
    756 static PtxSourceCache g_ptxSourceCache;
    757
    758 const char* getPtxString( const char* sample, const char* filename, const char** log )
    759 {
    760     if( log )
    761         *log = NULL;
    762 
    763     std::string *                                 ptx, cu;
    764     std::string                                   key  = std::string( filename ) + ";" + ( sample ? sample : "" );
    765     std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find( key );
    766 
    767     if( elem == g_ptxSourceCache.map.end() )
    768     {
    769         ptx = new std::string();
    770 #if CUDA_NVRTC_ENABLED
    771         std::string location;
    772         getCuStringFromFile( cu, location, sample, filename );
    773         getPtxFromCuString( *ptx, sample, cu.c_str(), location.c_str(), log );
    774 #else
    775         getPtxStringFromFile( *ptx, sample, filename );
    776 #endif
    777         g_ptxSourceCache.map[key] = ptx;
    778     }
    779     else
    780     {
    781         ptx = elem->second;
    782     }
    783 
    784     return ptx->c_str();
    785 }




::

    epsilon:SDK blyth$ optix7-fl getPtxString
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp
    ./optixCutouts/optixCutouts.cpp
    ./optixTriangle/optixTriangle.cpp
    ./optixMultiGPU/optixMultiGPU.cpp
    ./optixPathTracer/optixPathTracer.cpp
    ./optixDemandTextureAdvanced/optixDemandTexture.cpp
    ./optixHello/optixHello.cpp
    ./optixWhitted/optixWhitted.cpp
    ./optixDemandTexture/optixDemandTexture.cpp
    ./optixRaycasting/optixRaycasting.cpp
    ./sutil/Scene.cpp
    ./sutil/sutil.cpp
    ./sutil/sutil.h
    ./optixSphere/optixSphere.cpp
    epsilon:SDK blyth$ 


::
            
     670 void createModule( PathTracerState& state )
     671 {
     672     OptixModuleCompileOptions module_compile_options = {};
     673     module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
     674     module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
     675     module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
     676 
     677     state.pipeline_compile_options.usesMotionBlur        = false;
     678     state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
     679     state.pipeline_compile_options.numPayloadValues      = 2;
     680     state.pipeline_compile_options.numAttributeValues    = 2;
     681     state.pipeline_compile_options.exceptionFlags        =
     682         OPTIX_EXCEPTION_FLAG_NONE;
     683         //OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
     684         //OPTIX_EXCEPTION_FLAG_DEBUG;
     685     state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
     686 
     687     const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixPathTracer.cu" );
     688 
     689     char   log[2048];
     690     size_t sizeof_log = sizeof( log );
     691     OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
     692                 state.context,
     693                 &module_compile_options,
     694                 &state.pipeline_compile_options,
     695                 ptx.c_str(),
     696                 ptx.size(),
     697                 log,
     698                 &sizeof_log,
     699                 &state.ptx_module
     700                 ) );
     701 }


::

    epsilon:SDK blyth$ optix7-f OPTIX_SAMPLE_NAME
    ./optixSimpleMotionBlur/optixSimpleMotionBlur.cpp:    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixSimpleMotionBlur.cu" );
    ./optixCutouts/optixCutouts.cpp:    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixCutouts.cu" );
    ./optixTriangle/optixTriangle.cpp:            const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixTriangle.cu" );
    ./optixMultiGPU/optixMultiGPU.cpp:    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixMultiGPU.cu" );
    ./optixPathTracer/optixPathTracer.cpp:    const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixPathTracer.cu" );
    ./optixDemandTextureAdvanced/optixDemandTexture.cpp:    const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixDemandTexture.cu" );
    ./optixHello/optixHello.cpp:            const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "draw_solid_color.cu" );
    ./optixWhitted/optixWhitted.cpp:        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "geometry.cu" );
    ./optixWhitted/optixWhitted.cpp:        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "camera.cu" );
    ./optixWhitted/optixWhitted.cpp:        const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "shading.cu" );
    ./optixDemandTexture/optixDemandTexture.cpp:            const std::string ptx        = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixDemandTexture.cu" );
    ./optixRaycasting/optixRaycasting.cpp:    std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixRaycasting.cu" );
    ./optixSphere/optixSphere.cpp:            const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixSphere.cu" );

    ./CMakeLists.txt:    COMPILE_DEFINITIONS OPTIX_SAMPLE_NAME_DEFINE=${target_name})


    334   # rule that specifies this linkage.
    335   target_link_libraries( ${target_name}
    336     ${GLFW_LIB_NAME}
    337     imgui
    338     sutil_7_sdk
    339     )
    340 
    341   set_target_properties( ${target_name} PROPERTIES
    342     COMPILE_DEFINITIONS OPTIX_SAMPLE_NAME_DEFINE=${target_name})
    343 
    344   if( UNIX AND NOT APPLE )
    345     # Force using RPATH instead of RUNPATH on Debian
    346     target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
    347   endif()
    348 

    ./sutil/sutil.h:#define OPTIX_SAMPLE_NAME_STRINGIFY2(name) #name
    ./sutil/sutil.h:#define OPTIX_SAMPLE_NAME_STRINGIFY(name) OPTIX_SAMPLE_NAME_STRINGIFY2(name)
    ./sutil/sutil.h:#define OPTIX_SAMPLE_NAME OPTIX_SAMPLE_NAME_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)

    045 // Some helper macros to stringify the sample's name that comes in as a define
     46 #define OPTIX_SAMPLE_NAME_STRINGIFY2(name) #name
     47 #define OPTIX_SAMPLE_NAME_STRINGIFY(name) OPTIX_SAMPLE_NAME_STRINGIFY2(name)
     48 #define OPTIX_SAMPLE_NAME OPTIX_SAMPLE_NAME_STRINGIFY(OPTIX_SAMPLE_NAME_DEFINE)
     49 



