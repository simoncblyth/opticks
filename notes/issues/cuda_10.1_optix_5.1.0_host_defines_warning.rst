cuda_10.1_optix_5.1.0_host_defines_warning
===========================================


Many of the below warnings in CUDA 10.1 code when using OptiX 5.1.0

::

    [ 77%] Building CXX object tests/CMakeFiles/compactionTest.dir/compactionTest.cc.o
    In file included from /usr/local/OptiX_510/include/optixu/../internal/optix_datatypes.h:33:0,
                     from /usr/local/OptiX_510/include/optixu/optixu_math_namespace.h:57,
                     from /usr/local/OptiX_510/include/optix_world.h:71,
                     from /home/blyth/local/opticks/include/OptiXRap/OXPPNS.hh:13,
                     from /home/blyth/local/opticks/include/OptiXRap/OContext.hh:19,
                     from /home/blyth/opticks/okop/tests/dirtyBufferTest.cc:5:
    /usr/local/cuda-10.1/include/host_defines.h:54:2: warning: #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead." [-Wcpp]
     #warning "host_defines.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
      ^
    In file included from /usr/local/OptiX_510/include/optixu/../internal/optix_datatypes.h:33:0,





