OptiX_510_CUDA_10.1_many_internal_header_warnings
====================================================

All use of OptiX headers giving this warning.::

    [ 33%] Building NVCC ptx file OptiXRap_generated_intersect_analytic_test.cu.ptx
    [ 34%] Building NVCC ptx file OptiXRap_generated_OEventTest.cu.ptx
    In file included from /usr/local/OptiX_510/include/internal/optix_declarations.h:45:0,
                     from /usr/local/OptiX_510/include/optix_host.h:41,
                     from /usr/local/OptiX_510/include/optix_world.h:45,
                     from /home/blyth/opticks/optixrap/cu/dirtyBufferTest.cu:1:
    /usr/local/cuda-10.1/include/host_defines.h:54:2: warning: #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead." [-Wcpp]
     #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
      ^
    In file included from /usr/local/OptiX_510/include/internal/optix_declarations.h:45:0,
                     from /usr/local/OptiX_510/include/optix_host.h:41,
                     from /usr/local/OptiX_510/include/optix_world.h:45,
                     from /home/blyth/opticks/optixrap/cu/pinhole_camera.cu:1:
    /usr/local/cuda-10.1/include/host_defines.h:54:2: warning: #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead." [-Wcpp]
     #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
      ^



oxrap/OXRAP_PUSH.hh::

    #ifdef __clang__
    
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wshadow"
    #pragma clang diagnostic ignored "-Wunused-parameter"
    
    #elif defined(__GNUC__) || defined(__GNUG__)
    
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Woverloaded-virtual"
    // #pragma GCC diagnostic ignored "-Wcpp"
    // https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html
    
    
    #elif defined(_MSC_VER)
    
    #pragma warning(push)
    // nonstandard extension used: nameless struct/union  (from glm )
    //#pragma warning( disable : 4201 )
    // members needs to have dll-interface to be used by clients
    //#pragma warning( disable : 4251 )
    //
    #endif
    
    
    
Seems cannot control "#warning" messages via pragma ?   

* https://gcc.gnu.org/onlinedocs/gcc-4.8.4/gcc/Warning-Options.html

-Wno-cpp
    (C, Objective-C, C++, Objective-C++ and Fortran only)

    Suppress warning messages emitted by #warning directives. 



* https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html

    #pragma GCC diagnostic kind option

    Modifies the disposition of a diagnostic. Note that not all diagnostics are
    modifiable; at the moment only warnings (normally controlled by ‘-W…’) can be
    controlled, and not all of them. Use -fdiagnostics-show-option to determine
    which diagnostics are controllable and which option controls them. 



 
