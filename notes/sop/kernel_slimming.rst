notes/sop/kernel_slimming.rst
==============================

When adding headers that are included into optix kernels - it is
good to be aware of the impact on the size of the flattened kernel,
in order to avoid bloating it any more than is necessary.

The CSGOptiX CMake machinery reports the flattened kernel size.
Can use this to observe:

1. kernel source difference between Debug and Release builds
2. kernel source differnce from changing an impl, eg TMM impl in S4MultiLayrStack_select.h::

     01 #pragma once
      2 /**
      3 S4MultiLayrStack_select.h
      4 ==========================
      5
      6 Single header that picks which impl of TMM calculation to use
      7
      8 **/
      9
     10 #ifdef WITH_CUSTOM4
     11 //#include "C4MultiLayrStack.h"
     12 #include "S4MultiLayrStack.h"
     13 #endif
     14
     15


Things to look for in the flattened Release kernel source
-----------------------------------------------------------

1. printf : THERE SHOULD BE NONE : ALL printf SHOULD BE HIDDEN BEHIND DEBUG MACROS ONLY ENABLED IN Debug BUILD

::

   grep printf /data1/blyth/local/opticks_Release_g411/build/CSGOptiX/CSGOptiX7_flattened.ii


2. look for code that is not needed on device - and macro such away

   * stray inclusion of unused headers is very common - remove them


::

    [lo] A[blyth@localhost sysrap]$ opticks-f tcomplex.h
    ./qudarap/qsim.h:#include "tcomplex.h"
    ./sysrap/tcomplex.h:tcomplex.h
    ./sysrap/tests/tcomplexTest.cu:#include "tcomplex.h"
    ./sysrap/tests/tcomplex_test.cc:#include "tcomplex.h"
    ./sysrap/CMakeLists.txt:    tcomplex.h
    [lo] A[blyth@localhost opticks]$


::

    FLATTENED_OUTPUT_FILE=/data1/blyth/local/opticks_Release_g411/build/CSGOptiX/CSGOptiX7_flattened.ii
    grep '^# [0-9]\\+ \+\".*\"' '${FLATTENED_OUTPUT_FILE}'


Examples using S4 TMM : see sysrap/S4MultiLayrStack_select.h : about 70k lines
-------------------------------------------------------------------------------

* thrust/complex.h is bloating the kernel by 120k lines compared to scomplex.h - BUT DO NOT NOTICE SIGNIFICANT PERFORMANCE DIFF


Release with S4 TMM::

    --------------------------------------------------
    [Monitoring] Flattened Kernel Path: /data1/blyth/local/opticks_Release_g411/build/CSGOptiX/CSGOptiX7_flattened.ii
     KERNEL_NVCC_INC_ARGS -I/cvmfs/opticks.ihep.ac.cn/external/OptiX_910/include;-I/home/blyth/opticks/CSGOptiX//usr/local/cuda-13.1/targets/x86_64-linux/include;-I/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl;-I/data1/blyth/local/opticks_Release_g411/include/SysRap;-I/data1/blyth/local/opticks_Release_g411/include/CSG;-I/data1/blyth/local/opticks_Release_g411/include/QUDARap;-I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/custom4/0.1.8/include/Custom4
     KERNEL_NVCC_DEF_ARGS -DCONFIG_Release;-DPRODUCTION;-DRNG_PHILOX;-DWITH_SIMULATE;-DWITH_SIMTRACE;-DWITH_RENDER;-DOPTICKS_CSGOPTIX;-DWITH_CUSTOM4;-DWITH_SER;-DWITH_THRUST;-D_LIBCUDACXX_HAS_NO_INT128=1;-DCCCL_DISABLE_INT128_SUPPORT=1
    [Monitoring] Total Line Count: 69911
    [Monitoring] Header Line Contribution Breakdown:
    Lines      Header / Source File
    ------------------------------------------------------------------------------------------


Debug with S4 TMM::

    --------------------------------------------------
    [Monitoring] Flattened Kernel Path: /data1/blyth/local/opticks_Debug_g411/build/CSGOptiX/CSGOptiX7_flattened.ii
     KERNEL_NVCC_INC_ARGS -I/cvmfs/opticks.ihep.ac.cn/external/OptiX_910/include;-I/home/blyth/opticks/CSGOptiX//usr/local/cuda-13.1/targets/x86_64-linux/include;-I/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl;-I/data1/blyth/local/opticks_Debug_g411/include/SysRap;-I/data1/blyth/local/opticks_Debug_g411/include/CSG;-I/data1/blyth/local/opticks_Debug_g411/include/QUDARap;-I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/custom4/0.1.8/include/Custom4
     KERNEL_NVCC_DEF_ARGS -DCONFIG_Debug;-DDEBUG_PIDX;-DDEBUG_PIDXYZ;-DRNG_PHILOX;-DWITH_SIMULATE;-DWITH_SIMTRACE;-DWITH_RENDER;-DOPTICKS_CSGOPTIX;-DWITH_CUSTOM4;-DWITH_SER;-DWITH_THRUST;-D_LIBCUDACXX_HAS_NO_INT128=1;-DCCCL_DISABLE_INT128_SUPPORT=1
    [Monitoring] Total Line Count: 70866
    --------------------------------------------------




Examples using C4 TMM : see sysrap/S4MultiLayrStack_select.h : about 187k lines
---------------------------------------------------------------------------------


Release with traditional C4 TMM::

    --------------------------------------------------
    [Monitoring] Flattened Kernel Path: /data1/blyth/local/opticks_Release_g411/build/CSGOptiX/CSGOptiX7_flattened.ii
     KERNEL_NVCC_INC_ARGS -I/cvmfs/opticks.ihep.ac.cn/external/OptiX_910/include;-I/home/blyth/opticks/CSGOptiX//usr/local/cuda-13.1/targets/x86_64-linux/include;-I/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl;-I/data1/blyth/local/opticks_Release_g411/include/SysRap;-I/data1/blyth/local/opticks_Release_g411/include/CSG;-I/data1/blyth/local/opticks_Release_g411/include/QUDARap;-I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/custom4/0.1.8/include/Custom4
     KERNEL_NVCC_DEF_ARGS -DCONFIG_Release;-DPRODUCTION;-DRNG_PHILOX;-DWITH_SIMULATE;-DWITH_SIMTRACE;-DWITH_RENDER;-DOPTICKS_CSGOPTIX;-DWITH_CUSTOM4;-DWITH_SER;-DWITH_THRUST;-D_LIBCUDACXX_HAS_NO_INT128=1;-DCCCL_DISABLE_INT128_SUPPORT=1
    [Monitoring] Total Line Count: 187167
    --------------------------------------------------


Debug with traditional C4 TMM::

    --------------------------------------------------
    [Monitoring] Flattened Kernel Path: /data1/blyth/local/opticks_Debug_g411/build/CSGOptiX/CSGOptiX7_flattened.ii
     KERNEL_NVCC_INC_ARGS -I/cvmfs/opticks.ihep.ac.cn/external/OptiX_910/include;-I/home/blyth/opticks/CSGOptiX//usr/local/cuda-13.1/targets/x86_64-linux/include;-I/usr/local/cuda-13.1/targets/x86_64-linux/include/cccl;-I/data1/blyth/local/opticks_Debug_g411/include/SysRap;-I/data1/blyth/local/opticks_Debug_g411/include/CSG;-I/data1/blyth/local/opticks_Debug_g411/include/QUDARap;-I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc15/Release/J25.7.3/ExternalLibs/custom4/0.1.8/include/Custom4
     KERNEL_NVCC_DEF_ARGS -DCONFIG_Debug;-DDEBUG_PIDX;-DDEBUG_PIDXYZ;-DRNG_PHILOX;-DWITH_SIMULATE;-DWITH_SIMTRACE;-DWITH_RENDER;-DOPTICKS_CSGOPTIX;-DWITH_CUSTOM4;-DWITH_SER;-DWITH_THRUST;-D_LIBCUDACXX_HAS_NO_INT128=1;-DCCCL_DISABLE_INT128_SUPPORT=1
    [Monitoring] Total Line Count: 187824
    --------------------------------------------------


