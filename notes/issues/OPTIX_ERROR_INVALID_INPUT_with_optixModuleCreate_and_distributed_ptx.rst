OPTIX_ERROR_INVALID_INPUT_with_optixModuleCreate_and_distributed_ptx
=======================================================================

Overview
---------

1. find that PTX compiled on workstation (sm_89) fails to runtime compile on older GPU on cluster (sm_70)
   causing OPTIX_ERROR_INVALID_INPUT: Optix call optixModuleCreate within PIP.cc

2. first try changing COMPUTE_CAPABILITY from 89 to 70 on workstation, that looks like 
   it works from OptiX point of view : but causes issues for Thrust seeding of gensteps

   * looks like PTX generation for OptiX needs compute_70/sm_70 to work on both workstation and cluster
   * other CUDA compilation needs multiple gencode to support at least 70 and 89  

3. this issue of some nvcc compilation options needed OptiX and others for Thrust motivated
   the leap to CMake LANGUAGES CUDA support and specifically use of CUDA_ARCHITECTURES
   which gives more control over options than the old global CUDA_NVCC_FLAGS

   * now using OPTICKS_COMPUTE_CAPABILITY single integer envvar for OptiX PTX compilation 
   * now using OPTICKS_COMPUTE_ARCHITECTURES comma delimited envvar for general CUDA compilation
     which allows targetting multiple GPUs (google for CMake CUDA_ARCHITECTURES)
  

 
Versions that could cause problems for cross use
--------------------------------------------------

* compute capability   (<-- hoping this it the primary one)
* CUDA
* Driver
* OptiX  


PTX compiled and tested on workstation giving error on GPU cluster
---------------------------------------------------------------------

/hpcfs/juno/junogpu/blyth/j/ydy/slurm-2079651.out::


    0041 Fri Apr 18 20:08:25 2025
      42 +-----------------------------------------------------------------------------------------+
      43 | NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
      44 |-----------------------------------------+------------------------+----------------------+
      45 | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
      46 | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
      47 |                                         |                        |               MIG M. |
      48 |=========================================+========================+======================|
      49 |   0  Tesla V100-SXM2-32GB           On  |   00000000:1E:00.0 Off |                    0 |
      50 | N/A   31C    P0             42W /  300W |       1MiB /  32768MiB |      0%      Default |
      51 |                                         |                        |                  N/A |
      52 +-----------------------------------------+------------------------+----------------------+
      53 
      54 +-----------------------------------------------------------------------------------------+
      55 | Processes:                                                                              |
      56 |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
      57 |        ID   ID                                                               Usage      |
      58 |=========================================================================================|
      59 |  No running processes found                                                             |
      60 +-----------------------------------------------------------------------------------------+


    4797 == Buffer Memory Management ==
    4798 == Random Svc ==
    4799 == Root Writer ==
    4800  == PMTSimParamSvc ==
    4801 GENTOOL MODE:  gun
    4802 [(0, 0, 0)] None
    4803 3inch PMT type:  Tub3inchV3
    4804 WARNING: mac file run.mac does not exist.
    4805 Traceback (most recent call last):
    4806   File "/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.6/el9_amd64_gcc11/Wed/bin/tut_detsim.py", line 53, in <module>
    4807     juno_application.run()
    4808   File "/cvmfs/opticks.ihep.ac.cn/oj/releases/J25.3.0_Opticks-v0.3.6/el9_amd64_gcc11/Wed/python/Tutorial/JUNOApplication.py", line 224, in run
    4809     self.toptask.run()
    4810 RuntimeError: OPTIX_ERROR_INVALID_INPUT: Optix call 'optixModuleCreate( Ctx::context, &module_compile_options, &pipeline_compile_options, ptx.c_str(), ptx.size(), log, &sizeof_log, &module )' failed: /home/blyth/opticks/CSGOptiX/PIP.cc:296)
    4811 Log:
    4812 
    4813 
    4814 junotoptask:detsimiotask.terminate  WARN: invalid state tranform ((StartUp)) => ((EndUp))
    4815 junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    4816 junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO: All the collected process names:
    4817 junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: summaries:
    4818 junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: number of measurements:  *** Break *** segmentation violation
    4819 slurmstepd: error: *** JOB 2079651 ON gpu016 CANCELLED AT 2025-04-18T20:23:35 DUE TO TIME LIMIT ***


* curious that it doesnt die cleanly, hangs until killed by time limit 


Increase logging
--------------------

::

    knobs()
    {
       type $FUNCNAME 

       local exceptionFlags
       local debugLevel
       local optLevel

       #exceptionFlags=STACK_OVERFLOW   
       exceptionFlags=NONE

       debugLevel=DEFAULT
       #debugLevel=NONE
       #debugLevel=FULL    ## FULL now causes an exception with OptiX 7.5 Driver Version: 515.43.04  CUDA Version: 11.7

       optLevel=DEFAULT
       #optLevel=LEVEL_0
       #optLevel=LEVEL_3

       #export PIP__max_trace_depth=1
       export PIP__CreatePipelineOptions_exceptionFlags=$exceptionFlags # NONE/STACK_OVERFLOW/TRACE_DEPTH/USER/DEBUG
       export PIP__CreateModule_debugLevel=$debugLevel  # DEFAULT/NONE/MINIMAL/MODERATE/FULL   (DEFAULT is MINIMAL)
       export PIP__linkPipeline_debugLevel=$debugLevel  # DEFAULT/NONE/MINIMAL/MODERATE/FULL   
       export PIP__CreateModule_optLevel=$optLevel      # DEFAULT/LEVEL_0/LEVEL_1/LEVEL_2/LEVEL_3  

       env | grep PIP__ 


       rm /var/tmp/OptixCache_$USER/optix7cache.db    ## delete the cache to see the compilation output every time 
       export Ctx=INFO
       export PIP=INFO
       #export CSGOptiX=INFO
    }
    knobs




Up the logging to get more of a reason::

    4802 ]CSGImport::importPrim.dump_LVID:1 node.lvid 105 LVID -1 name uni1_0 soname uni1_0
    4803 sdirectory::DirList path /hpcfs/juno/junogpu/blyth/.opticks/rngcache/RNG pfx SCurandChunk_ ext .bin NO ENTRIES FOUND
    4804 2025-04-21 10:53:50.895 INFO  [1103421] [QRng::initStates@72] initStates<Philox> DO NOTHING : No LoadAndUpload needed  rngmax 1000000000 SEventConfig::MaxCurand 1000000000
    4805 2025-04-21 10:53:51.094 INFO  [1103421] [Ctx::log_cb@43] [ 4][       KNOBS]: All knobs on default.
    4806 
    4807 2025-04-21 10:53:51.321 INFO  [1103421] [Ctx::log_cb@43] [ 4][   DISKCACHE]: Opened database: "/var/tmp/OptixCache_blyth/optix7cache.db"
    4808 2025-04-21 10:53:51.322 INFO  [1103421] [Ctx::log_cb@43] [ 4][   DISKCACHE]:     Cache data size: "0 Bytes"
    4809 2025-04-21 10:53:51.360 INFO  [1103421] [PIP::CreateModule@254]
    4810  ptx_path /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-v0.3.6/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    4811  ptx.size 2055724
    4812  ptx_ok YES
    4813 
    4814 2025-04-21 10:53:51.360 INFO  [1103421] [PIP::CreateModule@275] [PIP::Desc
    4815  PIP__CreateModule_optLevel    DEFAULT
    4816  PIP__CreateModule_debugLevel  DEFAULT
    4817 ]PIP::Desc
    4818 [PIP::Desc_ModuleCompileOptions
    4819  module_compile_options.maxRegisterCount 0 OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0
    4820  module_compile_options.optLevel         0 OPTIX_COMPILE_OPTIMIZATION_DEFAULT
    4821  module_compile_options.debugLevel       0 OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT
    4822 ]PIP::Desc_ModuleCompileOptions
    4823 
    4824 2025-04-21 10:53:51.361 INFO  [1103421] [Ctx::log_cb@43] [ 4][   DISKCACHE]: Cache miss for key: ptx-2055724-key9a3f6a76be27617881503c5182f126d9-sm_70-rtc0-drv555.42.06
    4825 
    4826 2025-04-21 10:53:51.834 INFO  [1103421] [Ctx::log_cb@43] [ 2][    COMPILER]: COMPILE ERROR: Malformed input. See compile details for more information.
    4827 Error: Invalid target architecture. Maximum feasible for current context: sm_70, found: sm_89
    4828 


::

    Error: Invalid target architecture. Maximum feasible for current context: sm_70, found: sm_89



Look at the PTX : Its targetting too high : sm_89 
----------------------------------------------------

::

    U[blyth@lxlogin002 ydy]$ head -19 $OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-34097967
    // Cuda compilation tools, release 12.4, V12.4.131
    // Based on NVVM 7.0.1
    //

    .version 8.4
    .target sm_89
    .address_size 64

        // .globl   __raygen__rg_dummy
    .extern .func  (.param .b32 func_retval0) vprintf
    (
        .param .b64 vprintf_param_0,
        .param .b64 vprintf_param_1
    )
    ;
    U[blyth@lxlogin002 ydy]$ 




OPTICKS_COMPUTE_CAPABILITY
-----------------------------

+---------+---------------------------+------+ 
| machine |   GPU                     |  CC  |
+=========+===========================+======+
| P       |  TITAN RTX                | 70   |
+---------+---------------------------+------+  
| A       |  Ada 5000                 | 89   |
+---------+---------------------------+------+
| C       |  Tesla V100-SXM2-32GB     | 70   | 
+---------+---------------------------+------+



* https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf


What Compute Capability to target for the distributed PTX ? 
---------------------------------------------------------------

* https://forums.developer.nvidia.com/t/understanding-compute-capability/313577


dhart::

    th OptiX, if you’re compiling to PTX or OptiX-IR, you can use the compute
    capability for whatever the minimum GPU version you need to support is, and
    newer GPUs will work. For example, use 50 if you need Maxwell support, or 60
    for Pascal and beyond. This is detailed in the “Program Input” section of the
    “Pipeline” chapter in the OptiX Programming Guide: 



Note the following requirements for nvcc and nvrtc compilation:

The streaming multiprocessor (SM) target of the input OptiX program must be
less than or equal to the SM version of the GPU for which the module is
compiled.  To generate code for the minimum supported GPU (Maxwell), use
architecture targets for SM 5.0, for example, --gpu-architecture=compute_50.
Because OptiX rewrites the code internally, those targets will work on any
newer GPU as well.  CUDA Toolkits 10.2 and newer throw deprecation warnings for
SM 5.0 targets. These can be suppressed with the compiler option
-Wno-deprecated-gpu-targets.

If support for Maxwell GPUs is not required, you can use the next higher
GPU architecture target SM 6.0 (Pascal) to suppress these warnings.  Use
--machine=64 (-m64). Only 64-bit code is supported in OptiX.  Define the output
type with --optix-ir or --ptx. Do not compile to obj or cubin.

* https://raytracing-docs.nvidia.com/optix8/guide/index.html#program_pipeline_creation#program-input








Change OPTICKS_COMPUTE_CAPABILITY on build machine (A) and rebuild PTX
--------------------------------------------------------------------------

::


    A[blyth@localhost CSGOptiX]$ touch CSGOptiX7.cu
    A[blyth@localhost CSGOptiX]$ om
    === om-env : normal running
    === om-make-one : CSGOptiX        /home/blyth/opticks/CSGOptiX                                 /data1/blyth/local/opticks_Debug/build/CSGOptiX              
    [  2%] Building NVCC ptx file CSGOptiX_generated_CSGOptiX7.cu.ptx
    [ 36%] Built target CSGOptiX
    [ 41%] Built target CSGOptiXDescTest
    [ 46%] Built target CSGOptiXRMTest
    ...
    [ 90%] Built target CSGOptiXRenderTest
    [ 95%] Built target CSGOptiXTMTest
    [100%] Built target CSGOptiXSMTest
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /data1/blyth/local/opticks_Debug/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    -- Up-to-date: /data1/blyth/local/opticks_Debug/ptx/CSGOptiX_generated_Check.cu.ptx
    -- Up-to-date: /data1/blyth/local/opticks_Debug/include/CSGOptiX/CSGOptiX.h
    -- Up-to-date: /data1/blyth/local/opticks_Debug/include/CSGOptiX/CSGOPTIX_API_EXPORT.hh
    ...
    -- Up-to-date: /data1/blyth/local/opticks_Debug/lib/CSGOptiXRenderInteractiveTest
    -- Up-to-date: /data1/blyth/local/opticks_Debug/lib/CSGOptiXVersion
    -- Up-to-date: /data1/blyth/local/opticks_Debug/lib/CSGOptiXVersionTest
    -- Up-to-date: /data1/blyth/local/opticks_Debug/lib/CSGOptiXRenderTest
    -- Up-to-date: /data1/blyth/local/opticks_Debug/lib/ParamsTest
    A[blyth@localhost CSGOptiX]$ 


Touching is not enough::

    A[blyth@localhost CSGOptiX]$ head -19 /data1/blyth/local/opticks_Debug/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-34097967
    // Cuda compilation tools, release 12.4, V12.4.131
    // Based on NVVM 7.0.1
    //

    .version 8.4
    .target sm_89
    .address_size 64

        // .globl   __raygen__rg_dummy
    .extern .func  (.param .b32 func_retval0) vprintf
    (
        .param .b64 vprintf_param_0,
        .param .b64 vprintf_param_1
    )
    ;


* Need to re-conf as compute capability effect on compilation flags done at CMake level ?
* NOPE : NO CHANGE 


COMPUTE_CAPABILITY how to change
------------------------------------

::

    A[blyth@localhost opticks]$ opticks-fl COMPUTE_CAPABILITY
    ./CSGOptiX/CMakeLists.txt
    ./bin/OKTest_macOS_standalone.sh
    ./bin/opticks-setup-minimal.sh
    ./cmake/Modules/OpticksBuildOptions.cmake
    ./cmake/Modules/OpticksCUDAFlags.cmake
    ./cmake/Modules/inactive/DetectGPU.cmake
    ./cmake/Modules/include/helper_cuda_fallback/9.1/helper_cuda.h
    ./cmake/Modules/include/helper_cuda_fallback/9.2/helper_cuda.h
    ./examples/Standalone/standalone.bash
    ./examples/UseOKConf/CMakeLists.txt
    ./examples/UseOptiX7GeometryInstanced/CMakeLists.txt
    ./examples/UseOptiX7GeometryInstancedGAS/CMakeLists.txt
    ./examples/UseOptiX7GeometryInstancedGASComp/CMakeLists.txt
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/CMakeLists.txt
    ./examples/UseOptiX7GeometryModular/CMakeLists.txt
    ./examples/UseOptiX7GeometryStandalone/CMakeLists.txt
    ./examples/UseOptiXExample/UseOptiXExample.cc
    ./examples/UseOptiXFan/CMakeLists.txt
    ./examples/UseOptiXGeometryInstancedStandalone/CMakeLists.txt
    ./examples/UseOptiXGeometryStandalone/CMakeLists.txt
    ./examples/UseOptiXNoCMake/UseOptiX.cc
    ./examples/UseOptiXProgram/UseOptiXProgram.cc
    ./externals/glm.bash
    ./okconf/CMakeLists.txt
    ./okconf/OKConf.h
    ./okconf/go.sh
    ./okconf/OKConf.cc
    ./oldopticks.bash
    ./om.bash
    ./optixrap/OContext.cc
    ./optixrap/tests/UseOptiX.cc
    ./optixrap/tests/UseOptiXTest.cc
    ./opticks.bash
    A[blyth@localhost opticks]$ 





om-conf special casing for OKConf
-------------------------------------

::

    P[blyth@localhost okconf]$ t om-conf-one
    om-conf-one () 
    { 
        local arg=$1;
        local iwd=$(pwd);
        local name=$(basename ${iwd/tests});
        local sdir=$(om-sdir $name);
        local bdir=$(om-bdir $name);
        if [ "$arg" == "clean" ]; then
            echo $msg removed bdir $bdir as directed by clean argument;
            rm -rf $bdir;
        fi;
        if [ ! -d "$bdir" ]; then
            echo $msg bdir $bdir does not exist : creating it;
            mkdir -p $bdir;
        fi;
        cd $bdir;
        printf "%s %-15s %-60s %-60s \n" "$msg" $name $sdir $bdir;
        local rc=0;
        if [ "$name" == "okconf" ]; then
            om-cmake-okconf $sdir;
            rc=$?;
        else
            om-cmake $sdir;
            rc=$?;
        fi;
        return $rc
    }
    P[blyth@localhost okconf]$ 


::

    P[blyth@localhost okconf]$ t om-cmake-okconf
    om-cmake-okconf () 
    { 
        local sdir=$1;
        local bdir=$PWD;
        [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
        local rc;
        cmake $sdir -G "$(om-cmake-generator)" -DCMAKE_BUILD_TYPE=$(opticks-buildtype) -DOPTICKS_PREFIX=$(om-prefix) -DCMAKE_INSTALL_PREFIX=$(om-prefix) -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules -DOptiX_INSTALL_DIR=$(opticks-optix-prefix) -DCOMPUTE_CAPABILITY=$(opticks-compute-capability);
        rc=$?;
        return $rc
    }

    P[blyth@localhost okconf]$ t om-cmake
    om-cmake () 
    { 
        local sdir=$1;
        local bdir=$PWD;
        [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
        local rc;
        cmake $sdir -G "$(om-cmake-generator)" -DCMAKE_BUILD_TYPE=$(opticks-buildtype) -DOPTICKS_PREFIX=$(om-prefix) -DCMAKE_INSTALL_PREFIX=$(om-prefix) -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules;
        rc=$?;
        return $rc
    }
    P[blyth@localhost okconf]$ 


This means that to update compute capability need to:

1. change build env OPTICKS_COMPUTE_CAPABILITY
2. re-conf and rebuild : OKConf 

   * this generates /data1/blyth/local/opticks_Debug/lib64/cmake/okconf/okconf-config.cmake with TOPMATTER
     which sets COMPUTE_CAPABILITY




Hmm this doesnt update it::

    okconf
    om-conf
    om

    cx
    om

    head -16 

Try nuclear option::

   o
   om-clean
   om-conf
   oo
    
That does it::

    A[blyth@localhost opticks]$ head -19 $OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-34097967
    // Cuda compilation tools, release 12.4, V12.4.131
    // Based on NVVM 7.0.1
    //

    .version 8.4
    .target sm_70
    .address_size 64

        // .globl   __raygen__rg_dummy
    .extern .func  (.param .b32 func_retval0) vprintf
    (
        .param .b64 vprintf_param_0,
        .param .b64 vprintf_param_1
    )
        





lib64/cmake/okconf/okconf-config.cmake TOPMATTER
--------------------------------------------------

::

    A[blyth@localhost qudarap]$ cat  /data1/blyth/local/opticks_Debug/lib64/cmake/okconf/okconf-config.cmake

    # PROJECT_NAME OKConf
    # TOPMATTER

    ## OKConf generated TOPMATTER

    set(OptiX_INSTALL_DIR /cvmfs/opticks.ihep.ac.cn/external/OptiX_800)
    set(COMPUTE_CAPABILITY 70)

    if(OKConf_VERBOSE)
      message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OKConf_VERBOSE     : ${OKConf_VERBOSE} ")
      message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OptiX_INSTALL_DIR  : ${OptiX_INSTALL_DIR} ")
      message(STATUS "${CMAKE_CURRENT_LIST_FILE} : COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY} ")
    endif()

    include(OpticksCUDAFlags)
    ## see notes/issues/OpticksCUDAFlags.rst




cmake/Modules/OpticksCUDAFlags.cmake::

    080 if(NOT (COMPUTE_CAPABILITY LESS 30))
     81 
     82    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
     83    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
     84    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
     85 
     86    list(APPEND CUDA_NVCC_FLAGS "-std=${OPTICKS_CUDA_NVCC_DIALECT}")
     87    # https://github.com/facebookresearch/Detectron/issues/185
     88    # notes/issues/g4_1062_opticks_with_newer_gcc_for_G4OpticksTest.rst 
     89 
     90    list(APPEND CUDA_NVCC_FLAGS "-O2")
     91    #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
     92    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
     93 
     94    list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored ") 
     95    # notes/issues/glm_anno_warnings_with_gcc_831.rst 
     96 
     97    #list(APPEND CUDA_NVCC_FLAGS "-m64")
     98    #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
     99 
    100    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    101    set(CUDA_VERBOSE_BUILD OFF)
    102 
    103 endif()



CMake CUDA_NVCC_FLAGS
-----------------------

* https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768/3


Robert_Crovella::

    CMake went through a significant change in how it dealt with CUDA in the 3.8 - 3.12 timefra

    The set(CUDA_NVCC_FLAGS… syntax was part of the old (deprecated)
    methodology. The target_compile_options(… syntax is part of the new methodology
    (so called “first class language support”).


* https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA.cmake


target_compile_options COMPILE_LANGUAGE:CUDA
-------------------------------------------------

* https://stackoverflow.com/questions/54504253/how-to-add-more-than-one-cuda-gencode-using-modern-cmake-per-target

* https://gitlab.kitware.com/cmake/cmake/-/issues/19502


::

    cmake_minimum_required(VERSION 3.14)
    project(test LANGUAGES CUDA)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    add_library(foo SHARED main.cu)

    target_compile_options(foo PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_60,code=sm_60>"
      )
    target_compile_options(foo PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_52,code=sm_52 -gencode arch=compute_50,code=sm_50>"
      )



Try CMake CUDA LANGUAGE Approach
----------------------------------------


::

    A[blyth@localhost qudarap]$ cmake --help-policy CMP0104
    CMP0104
    -------

    .. versionadded:: 3.18

    Initialize ``CMAKE_CUDA_ARCHITECTURES`` when
    ``CMAKE_CUDA_COMPILER_ID`` is ``NVIDIA``.
    Raise an error if ``CUDA_ARCHITECTURES`` is empty.

    ``CMAKE_CUDA_ARCHITECTURES`` introduced in CMake 3.18 is used to
    initialize ``CUDA_ARCHITECTURES``, which passes correct code generation
    flags to the CUDA compiler.

    Previous to this users had to manually specify the code generation flags. This
    policy is for backwards compatibility with manually specifying code generation
    flags.

    The ``OLD`` behavior for this policy is to not initialize
    ``CMAKE_CUDA_ARCHITECTURES`` when
    ``CMAKE_CUDA_COMPILER_ID`` is ``NVIDIA``.
    Empty ``CUDA_ARCHITECTURES`` is allowed.

    The ``NEW`` behavior of this policy is to initialize
    ``CMAKE_CUDA_ARCHITECTURES`` when
    ``CMAKE_CUDA_COMPILER_ID`` is ``NVIDIA``
    and raise an error if ``CUDA_ARCHITECTURES`` is empty during generation.

    If ``CUDA_ARCHITECTURES`` is set to a false value no architectures
    flags are passed to the compiler. This is intended to support packagers and
    the rare cases where full control over the passed flags is required.

    This policy was introduced in CMake version 3.18.  CMake version
    3.26.5 warns when the policy is not set and uses ``OLD`` behavior.
    Use the ``cmake_policy()`` command to set it to ``OLD`` or ``NEW``
    explicitly.

    .. note::
      The ``OLD`` behavior of a policy is
      ``deprecated by definition``
      and may be removed in a future version of CMake.

    Examples
    ^^^^^^^^

     set_target_properties(tgt PROPERTIES CUDA_ARCHITECTURES "35;50;72")

    Generates code for real and virtual architectures ``30``, ``50`` and ``72``.

     set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)

    Generates code for real architecture ``70`` and virtual architecture ``72``.

     set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES OFF)

    CMake will not pass any architecture flags to the compiler.
    A[blyth@localhost qudarap]$ 






PIP logging from optixModuleCreate
-----------------------------------

::

    280 
    281     size_t sizeof_log = 0 ;
    282     char log[2048]; // For error reporting from OptiX creation functions
    283 
    284 #if OPTIX_VERSION <= 70600
    285     OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    286                 Ctx::context,
    287                 &module_compile_options,
    288                 &pipeline_compile_options,
    289                 ptx.c_str(),
    290                 ptx.size(),
    291                 log,
    292                 &sizeof_log,
    293                 &module
    294                 ) );
    295 #else
    296     OPTIX_CHECK_LOG( optixModuleCreate(
    297                 Ctx::context,
    298                 &module_compile_options,
    299                 &pipeline_compile_options,
    300                 ptx.c_str(),
    301                 ptx.size(),
    302                 log,
    303                 &sizeof_log,
    304                 &module
    305                 ) );
    306 
    307 #endif
    308 
    309     return module ;
    310 }





OPTIX_ERROR_INVALID_INPUT optixModuleCreate
--------------------------------------------

* https://forums.developer.nvidia.com/t/optix-error-optix-error-invalid-input/286602
* https://forums.developer.nvidia.com/t/optixmodulecreate-throws-error-compile-error-only-in-debug/301369


::

    P[blyth@localhost opticks]$ export PIP__CreateModule_debugLevel=FULL
    P[blyth@localhost opticks]$ cxr_min.sh 
    ...
    /data/blyth/opticks_Debug/bin/cxr_min.sh : run : delete prior LOG CSGOptiXRenderInteractiveTest.log
    2025-04-21 10:08:31.777 INFO  [302648] [SEventConfig::SetDevice@1333] SEventConfig::DescDevice
    name                             : NVIDIA TITAN RTX
    totalGlobalMem_bytes             : 25396576256
    totalGlobalMem_GB                : 23
    HeuristicMaxSlot(VRAM)           : 197276976
    HeuristicMaxSlot(VRAM)/M         : 197
    HeuristicMaxSlot_Rounded(VRAM)   : 197000000
    MaxSlot/M                        : 0

    2025-04-21 10:08:31.778 INFO  [302648] [SEventConfig::SetDevice@1345]  Configured_MaxSlot/M 0 Final_MaxSlot/M 197 HeuristicMaxSlot_Rounded/M 197 changed YES DeviceName NVIDIA TITAN RTX HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_INVALID_VALUE: Optix call 'optixModuleCreateFromPTX( Ctx::context, &module_compile_options, &pipeline_compile_options, ptx.c_str(), ptx.size(), log, &sizeof_log, &module )' failed: /home/blyth/opticks/CSGOptiX/PIP.cc:285)
    Log:
    P�S

    /data/blyth/opticks_Debug/bin/cxr_min.sh: line 245: 302648 Aborted                 (core dumped) $bin
    /data/blyth/opticks_Debug/bin/cxr_min.sh run error
    P[blyth@localhost opticks]$ 






    284 #if OPTIX_VERSION <= 70600
    285     OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
    286                 Ctx::context,
    287                 &module_compile_options,
    288                 &pipeline_compile_options,
    289                 ptx.c_str(),
    290                 ptx.size(),
    291                 log,
    292                 &sizeof_log,
    293                 &module
    294                 ) );
    295 #else
    296     OPTIX_CHECK_LOG( optixModuleCreate(
    297                 Ctx::context,
    298                 &module_compile_options,
    299                 &pipeline_compile_options,
    300                 ptx.c_str(),
    301                 ptx.size(),
    302                 log,
    303                 &sizeof_log,
    304                 &module
    305                 ) );
    306 
    307 #endif
    308 



distributing PTX
--------------------

* https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/


PTX is similar to LLVM IR in that the PTX representation of a program can be
compiled to a wide range of NVIDIA GPUs. Importantly, this compilation of PTX
for a specific GPU can happen just-in-time (JIT) at application runtime. As
shown in Figure 1, the executable for an application can embed both GPU
binaries (cubins) and PTX code. Embedding the PTX in the executable enables
CUDA to JIT compile the PTX to the appropriate cubin at application runtime.
The JIT compiler for PTX is part of the NVIDIA GPU driver. 

Embedding PTX in the application enables running the first stage of
compilation—high-level language to PTX—when the application is compiled. The
second stage of compilation—PTX to cubin—can be delayed until application
runtime. As illustrated below, doing this allows the application to run on a
wider range of GPUs, including GPUs released well after the application was
built. 

Compute capability
~~~~~~~~~~~~~~~~~~~

* https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/

All NVIDIA GPUs have a version identifier known as the compute capability, or
CC number. Each compute capability has a major and a minor version number. For
example, compute capability 8.6 has a major version of 8 and a minor version of
6. 

Like any processor, NVIDIA GPUs have a specific ISA. GPUs from different
generations have different ISAs. These ISAs are identified by a version number
which corresponds to the GPU’s compute capability. When a binary (cubin) is
compiled, it is compiled for a specific compute capability. 

For example, GeForce and RTX GPUs from the NVIDIA Ampere generation have a
compute capability of 8.6 and their cubin version is sm_86. All cubin versions
have the format sm_XY where X and Y correspond to the major and minor numbers
of a compute capability.

NVIDIA GPUs of different generations and even different products within a
generation can have different ISAs. This is part of the reason for having PTX.


PTX JIT compatibility
~~~~~~~~~~~~~~~~~~~~~

* https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/

Embedding PTX in an executable provides a mechanism for compatibility across
GPUs of different compute capabilities, including different major versions,
within a single binary file. As illustrated in the executable in Figure 1, both
PTX and cubin can be stored in the final application executable. PTX and cubin
can also be stored in libraries. 

When the PTX code is stored in an application or library binary, it can be JIT
compiled for the GPU it is being loaded on. For example, if the application or
library contains PTX targeting compute_70, that PTX can be JIT compiled for any
GPU of compute capability 7.0 or higher, including compute capability 8.x, 9.x,
10.x, and 12.x. 

PTX cannot be JIT compiled for compute capabilities lower than the PTX version.
For example,  PTX targeting compute_70 cannot be JIT compiled for a compute
capability 5.x or 6.x GPU. 






After target change from 89 to 70 for OptiX PTX get three thrust-qudarap related test fails
-----------------------------------------------------------------------------------------------

* it is as if the CUDA compilation options needed for OptiX to target older GPU
  do not work with thrust on the newer GPU


::

    FAILS:  3   / 217   :  Mon Apr 21 15:03:50 2025   
      10 /21  Test #10 : QUDARapTest.QEventTest                        ***Failed                      0.69   
      11 /21  Test #11 : QUDARapTest.QEvent_Lifecycle_Test             ***Failed                      0.43   
      12 /21  Test #12 : QUDARapTest.QSimWithEventTest                 ***Failed                      2.93   


::

    A[blyth@localhost tests]$ TEST=one QEventTest
    2025-04-21 15:17:27.756 INFO  [1360165] [SEventConfig::SetDevice@1333] SEventConfig::DescDevice
    name                             : NVIDIA RTX 5000 Ada Generation
    totalGlobalMem_bytes             : 33796980736
    totalGlobalMem_GB                : 31
    HeuristicMaxSlot(VRAM)           : 262530128
    HeuristicMaxSlot(VRAM)/M         : 262
    HeuristicMaxSlot_Rounded(VRAM)   : 262000000
    MaxSlot/M                        : 0

    2025-04-21 15:17:27.756 INFO  [1360165] [SEventConfig::SetDevice@1345]  Configured_MaxSlot/M 0 Final_MaxSlot/M 262 HeuristicMaxSlot_Rounded/M 262 changed YES DeviceName NVIDIA RTX 5000 Ada Generation HasDevice YES
    (export OPTICKS_MAX_SLOT=0 # to use VRAM based HeuristicMaxPhoton) 
    QEventTest::setGenstep_one
    terminate called after throwing an instance of 'thrust::THRUST_200302_700_NS::system::system_error'
      what():  after reduction step 1: cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
    Aborted (core dumped)
    A[blyth@localhost tests]$ 


Add some debug to find where the error happens::

    2025-04-21 15:22:19.741 INFO  [1360840] [QEventTest::setGenstep_one@93] [ QEvent::setGenstepUpload_NP 
    terminate called after throwing an instance of 'thrust::THRUST_200302_700_NS::system::system_error'
      what():  after reduction step 1: cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
    Aborted (core dumped)


::

     170 /**
     171 QEvent::setGenstepUpload_NP
     172 ------------------------------
     173 
     174 Canonically invoked from QSim::simulate and QSim::simtrace just prior to cx->launch 
     175 
     176 **/
     177 int QEvent::setGenstepUpload_NP(const NP* gs_ )
     178 {
     179     LOG_IF(info, SEvt::LIFECYCLE) << "[" ;
     180     int rc = setGenstepUpload_NP(gs_, nullptr );
     181     LOG_IF(info, SEvt::LIFECYCLE) << "]" ;
     182     return rc ;
     183 }


Add some more to logging QEvent::

    A[blyth@localhost qudarap]$ TEST=one VERBOSE=1 QEvent=INFO QEventTest
    ...
    2025-04-21 15:31:53.799 INFO  [1361643] [QEventTest::setGenstep_one@93] [ QEvent::setGenstepUpload_NP 
    2025-04-21 15:31:53.799 INFO  [1361643] [QEvent::setGenstepUpload_NP@195]  gs (9, 6, 4, )SGenstep::DescGensteps num_genstep 9 (3 5 2 0 1 3 4 2 4 ) total 24
    2025-04-21 15:31:53.799 INFO  [1361643] [QEvent::setGenstepUpload@327]  gs_start 0 gs_stop 9 evt.num_genstep 9 not_allocated YES zero_genstep NO 
    2025-04-21 15:31:53.799 INFO  [1361643] [QEvent::setGenstepUpload@337] [ device_alloc_genstep_and_seed 
    2025-04-21 15:31:53.799 INFO  [1361643] [QEvent::device_alloc_genstep_and_seed@432]  device_alloc genstep and seed 
    2025-04-21 15:31:53.800 INFO  [1361643] [QEvent::setGenstepUpload@339] ] device_alloc_genstep_and_seed 
    2025-04-21 15:31:53.800 INFO  [1361643] [QEvent::setGenstepUpload@353] [ QU::copy_host_to_device 
    2025-04-21 15:31:53.800 INFO  [1361643] [QEvent::setGenstepUpload@355] ] QU::copy_host_to_device 
    2025-04-21 15:31:53.800 INFO  [1361643] [QEvent::setGenstepUpload@362] [ QU::device_memset 
    2025-04-21 15:31:53.800 INFO  [1361643] [QEvent::setGenstepUpload@364] ] QU::device_memset 
    2025-04-21 15:31:53.800 INFO  [1361643] [QEvent::setGenstepUpload@374] [ count_genstep_photons_and_fill_seed_buffer 
    terminate called after throwing an instance of 'thrust::THRUST_200302_700_NS::system::system_error'
      what():  after reduction step 1: cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
    Aborted (core dumped)



::

     570 extern "C" void QEvent_count_genstep_photons_and_fill_seed_buffer(sevent* evt );
     571 void QEvent::count_genstep_photons_and_fill_seed_buffer()
     572 {
     573     LOG_IF(info, LIFECYCLE) ;
     574     QEvent_count_genstep_photons_and_fill_seed_buffer( evt );
     575 }



::

    181 extern "C" void QEvent_count_genstep_photons_and_fill_seed_buffer(sevent* evt )
    182 {
    183     typedef typename thrust::device_vector<int>::iterator Iterator;
    184 
    185     thrust::device_ptr<int> t_gs = thrust::device_pointer_cast( (int*)evt->genstep ) ;
    186 
    187 #ifdef DEBUG_QEVENT
    188     printf("//QEvent_count_genstep_photons sevent::genstep_numphoton_offset %d  sevent::genstep_itemsize  %d  \n",
    189             sevent::genstep_numphoton_offset, sevent::genstep_itemsize );
    190 #endif
    191 
    192 
    193     strided_range<Iterator> gs_pho(
    194         t_gs + sevent::genstep_numphoton_offset,
    195         t_gs + evt->num_genstep*sevent::genstep_itemsize ,
    196         sevent::genstep_itemsize );    // begin, end, stride 
    197 
    198     evt->num_seed = thrust::reduce(gs_pho.begin(), gs_pho.end() );
    199 
    200 #ifdef DEBUG_QEVENT
    201     printf("//QEvent_count_genstep_photons_and_fill_seed_buffer evt.num_genstep %d evt.num_seed %d evt.max_photon %d \n", evt->num_genstep, evt->num_seed, evt->max_photon );
    202 #endif
    203 
    204     bool expect_seed =  evt->seed && evt->num_seed > 0 ;
    205     if(!expect_seed) printf("//QEvent_count_genstep_photons_and_fill_seed_buffer  evt.seed %s  evt.num_seed %d \n",  (evt->seed ? "YES" : "NO " ), evt->num_seed );
    206     assert( expect_seed );
    ...






OpticksCUDAFlags.cmake::

    080 if(NOT (COMPUTE_CAPABILITY LESS 30))
     81 
     82    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
     83    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
     84    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
     85 
     86    list(APPEND CUDA_NVCC_FLAGS "-std=${OPTICKS_CUDA_NVCC_DIALECT}")
     87    # https://github.com/facebookresearch/Detectron/issues/185
     88    # notes/issues/g4_1062_opticks_with_newer_gcc_for_G4OpticksTest.rst 
     89 
     90    list(APPEND CUDA_NVCC_FLAGS "-O2")
     91    #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
     92    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
     93 
     94    list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored ")
     95    # notes/issues/glm_anno_warnings_with_gcc_831.rst 
     96 
     97    #list(APPEND CUDA_NVCC_FLAGS "-m64")
     98    #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
     99 
    100    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    101    set(CUDA_VERBOSE_BUILD OFF)
    102 
    103 endif()


Building QUDARap : nvcc compiles into .o
-----------------------------------------

::

    === om-make-one : qudarap         /home/blyth/opticks/qudarap                                  /data1/blyth/local/opticks_Debug/build/qudarap               
    [  1%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QTexLookup.cu.o
    [  2%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QBnd.cu.o
    [  3%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QCerenkov.cu.o
    [  4%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QCurandState.cu.o
    [  5%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QCurandStateMonolithic.cu.o
    [  6%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QEvent.cu.o
    [  8%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QMultiFilm.cu.o
    [ 10%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QPoly.cu.o
    [ 10%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QOptical.cu.o
    [ 11%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QPMT.cu.o
    [ 12%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QProp.cu.o
    [ 13%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QRng.cu.o
    [ 15%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QScint.cu.o
    [ 17%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QTex.cu.o
    [ 17%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QSim.cu.o
    [ 23%] Building CXX object CMakeFiles/QUDARap.dir/QCurandStateMonolithic.cc.o
    [ 23%] Building CXX object CMakeFiles/QUDARap.dir/QPMT.cc.o
    [ 23%] Building CXX object CMakeFiles/QUDARap.dir/QUDARAP_LOG.cc.o


::

    A[blyth@localhost lib64]$ nm -D --defined-only libQUDARap.so | wc -l 
    1002


    A[blyth@localhost lib64]$ nm -D --defined-only libQUDARap.so | c++filt  | grep QEvent_
    00000000000e9680 T QEvent_checkEvt
    00000000000e97f0 T QEvent_count_genstep_photons
    00000000000e9990 T QEvent_count_genstep_photons_and_fill_seed_buffer
    00000000000e98a0 T QEvent_fill_seed_buffer
    00000000000e9670 T _QEvent_checkEvt(sevent*, unsigned int, unsigned int)
    00000000000e95d0 T __device_stub__Z16_QEvent_checkEvtP6seventjj(sevent*, unsigned int, unsigned int)
    A[blyth@localhost lib64]$ 



CSGOptiX flags for PTX
------------------------

::

    A[blyth@localhost CSGOptiX]$ touch CSGOptiX7.cu 
    A[blyth@localhost CSGOptiX]$ VERBOSE=1 om


    -- Generating /data1/blyth/local/opticks_Debug/build/CSGOptiX/CSGOptiX_generated_CSGOptiX7.cu.ptx
    /usr/local/cuda-12.4/bin/nvcc 
          /home/blyth/opticks/CSGOptiX/CSGOptiX7.cu 
          -ptx 
          -o /data1/blyth/local/opticks_Debug/build/CSGOptiX/CSGOptiX_generated_CSGOptiX7.cu.ptx 
          -ccbin /usr/bin/cc 
          -m64 
           -DWITH_PRD 
           -DWITH_SIMULATE 
           -DWITH_SIMTRACE 
           -DWITH_RENDER 
           -DOPTICKS_CSGOPTIX 
           -DWITH_THRUST 
           -DOPTICKS_CSG 
           -DWITH_CONTIGUOUS 
           -DWITH_S_BB 
           -DWITH_CUSTOM4 
           -DCONFIG_Debug -DOPTICKS_SYSRAP -DWITH_CHILD -DPLOG_LOCAL -DRNG_PHILOX -DDEBUG_PIDX -DDEBUG_PIDXYZ -DWITH_STTF -DWITH_SLOG -DOPTICKS_OKCONF -DOPTICKS_QUDARAP 
           -Xcompiler 
           -fPIC 
           -gencode=arch=compute_70,code=sm_70 
           -std=c++17 
           -O2 
           --use_fast_math 
           -Xcudafe 
           --diag_suppress=esa_on_defaulted_function_ignored 
          -DNVCC 
          -I/usr/local/cuda-12.4/include 
          -I/data1/blyth/local/opticks_Debug/externals/glm/glm 
          -I/cvmfs/opticks.ihep.ac.cn/external/OptiX_800/include 
          -I/home/blyth/opticks/CSGOptiX 
          -I/data1/blyth/local/opticks_Debug/include/CSG 
          -I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/custom4/0.1.8/include/Custom4 
          -I/data1/blyth/local/opticks_Debug/include/SysRap 
          -I/data1/blyth/local/opticks_Debug/externals/plog/include 
          -I/data1/blyth/local/opticks_Debug/include/OKConf 
          -I/data1/blyth/local/opticks_Debug/externals/include/nljson 
          -I/data1/blyth/local/opticks_Debug/include/QUDARap
    Generated /data1/blyth/local/opticks_Debug/build/CSGOptiX/CSGOptiX_generated_CSGOptiX7.cu.ptx successfully.


QUDARap flags for QEvent.cu
-----------------------------

::

    A[blyth@localhost qudarap]$ touch QEvent.cu
    A[blyth@localhost qudarap]$ VERBOSE=1 om


      1%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QEvent.cu.o
    cd /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir && /usr/bin/cmake -E make_directory /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//.

    cd /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir && /usr/bin/cmake -D verbose:BOOL=1 -D build_configuration:STRING=Debug -D generated_file:STRING=/data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o -D generated_cubin_file:STRING=/data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o.cubin.txt -P /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.Debug.cmake

    -- Removing /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o
    /usr/bin/cmake -E rm -f /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o

    -- Generating dependency file: /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.NVCC-depend
    /usr/local/cuda-12.4/bin/nvcc -M -D__CUDACC__ /home/blyth/opticks/qudarap/QEvent.cu -o /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.NVCC-depend -ccbin /usr/bin/cc -m64 -DQUDARap_EXPORTS -DWITH_CUSTOM4 -DOPTICKS_QUDARAP -DWITH_THRUST -DCONFIG_Debug -DOPTICKS_SYSRAP -DWITH_CHILD -DPLOG_LOCAL -DRNG_PHILOX -DDEBUG_PIDX -DDEBUG_PIDXYZ -DWITH_STTF -DWITH_SLOG -DOPTICKS_OKCONF -Xcompiler ,\"-fPIC\" -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70 -std=c++17 -O2 --use_fast_math -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -DNVCC -I/usr/local/cuda-12.4/include -I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/custom4/0.1.8/include/Custom4 -I/home/blyth/opticks/qudarap -I/data1/blyth/local/opticks_Debug/externals/glm/glm -I/data1/blyth/local/opticks_Debug/include/SysRap -I/data1/blyth/local/opticks_Debug/externals/plog/include -I/data1/blyth/local/opticks_Debug/include/OKConf -I/data1/blyth/local/opticks_Debug/externals/include/nljson

    -- Generating temporary cmake readable file: /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend.tmp
    /usr/bin/cmake -D input_file:FILEPATH=/data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.NVCC-depend -D output_file:FILEPATH=/data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend.tmp -D verbose=1 -P /usr/share/cmake/Modules/FindCUDA/make2cmake.cmake

    -- Copy if different /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend.tmp to /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend
    /usr/bin/cmake -E copy_if_different /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend.tmp /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend

    -- Removing /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend.tmp and /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.NVCC-depend
    /usr/bin/cmake -E rm -f /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.depend.tmp /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//QUDARap_generated_QEvent.cu.o.NVCC-depend

    -- Generating /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o
    /usr/local/cuda-12.4/bin/nvcc 
          /home/blyth/opticks/qudarap/QEvent.cu 
         -c 
         -o /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o 
         -ccbin /usr/bin/cc 
         -m64 
         -DQUDARap_EXPORTS 
         -DWITH_CUSTOM4 -DOPTICKS_QUDARAP -DWITH_THRUST -DCONFIG_Debug -DOPTICKS_SYSRAP -DWITH_CHILD -DPLOG_LOCAL -DRNG_PHILOX -DDEBUG_PIDX -DDEBUG_PIDXYZ -DWITH_STTF -DWITH_SLOG -DOPTICKS_OKCONF 
         -Xcompiler ,\"-fPIC\" 
         -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70 
         -std=c++17 
         -O2
         --use_fast_math 
         -Xcudafe 
         --diag_suppress=esa_on_defaulted_function_ignored 
         -DNVCC 
         -I/usr/local/cuda-12.4/include
         -I/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Pre-Release/J24.1.x/ExternalLibs/custom4/0.1.8/include/Custom4
         -I/home/blyth/opticks/qudarap
         -I/data1/blyth/local/opticks_Debug/externals/glm/glm
         -I/data1/blyth/local/opticks_Debug/include/SysRap
         -I/data1/blyth/local/opticks_Debug/externals/plog/include
         -I/data1/blyth/local/opticks_Debug/include/OKConf
         -I/data1/blyth/local/opticks_Debug/externals/include/nljson
    Generated /data1/blyth/local/opticks_Debug/build/qudarap/CMakeFiles/QUDARap.dir//./QUDARap_generated_QEvent.cu.o successfully



-gencode=arch=compute_70,code=sm_70
----------------------------------------

* https://kaixih.github.io/nvcc-options/


-arch
    specifies which virtual compute architecture the PTX code should be generated against.
    The valid format is like: -arch=compute_XY.
    (relevant to first stage .cu -> .ptx)

-code
    specifies which actual sm architecture the SASS code should be generated against
    and be included in the binary. The valid format is like: -code=sm_XY
    (relevant to second stage .ptx -> .sass)

-gencode
    combines both -arch and -code.
    The valid format is like: -gencode=arch=compute_XY,code=sm_XY


-arch=compute_Xa is compatible with -code=sm_Xb when a≤b.



* https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

When you compile CUDA code, you should always compile only one ‘-arch‘ flag
that matches your most used GPU cards. This will enable faster runtime, because
code generation will occur during compilation.  If you only mention ‘-gencode‘,
but omit the ‘-arch‘ flag, the GPU code generation will occur on the JIT
compiler by the CUDA driver.

When you want to speed up CUDA compilation, you want to reduce the amount of
irrelevant ‘-gencode‘ flags. However, sometimes you may wish to have better
CUDA backwards compatibility by adding more comprehensive ‘-gencode‘ flags.

Before you continue, identify which GPU you have and which CUDA version you
have installed first.

Lots of gencode examples eg::

    Sample flags for generation on CUDA 11.7 for maximum compatibility with V100 (sm_70)
    and T4 Turing Datacenter cards (sm_75), but also support newer RTX 3080 (sm_86), 
    and Drive AGX Orin (sm_87):

    -arch=sm_52 \
    -gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_87,code=sm_87
    -gencode=arch=compute_86,code=compute_86



CUDA: How to use -arch and -code and SM vs COMPUTE
----------------------------------------------------

* https://stackoverflow.com/questions/35656294/cuda-how-to-use-arch-and-code-and-sm-vs-compute



cuobjdump
------------

::


    A[blyth@localhost qudarap]$ cuobjdump -ltext $OPTICKS_PREFIX/lib64/libQUDARap.so 
    SASS text section 1 : x-_Z35_QCurandStateMonolithic_curand_initiiP11qcurandwrapI17curandStateXORWOWEPS0_.sm_70.elf.bin
    SASS text section 2 : x-_Z31_QCurandState_curand_init_chunkiiP10scurandrefI17curandStateXORWOWEPS0_.sm_70.elf.bin
    SASS text section 3 : x-_Z21_QSim_prop_lookup_oneIfEvP4qsimPT_PKS2_jjjj.sm_70.elf.bin
    SASS text section 4 : x-_Z21_QSim_prop_lookup_oneIdEvP4qsimPT_PKS2_jjjj.sm_70.elf.bin
    SASS text section 5 : x-_Z17_QSim_prop_lookupIfEvP4qsimPT_PKS2_jPjj.sm_70.elf.bin
    SASS text section 6 : x-_Z17_QSim_prop_lookupIdEvP4qsimPT_PKS2_jPjj.sm_70.elf.bin
    SASS text section 7 : x-_Z18_QSim_rng_sequenceIdEvP4qsimPT_jjj.sm_70.elf.bin
    SASS text section 8 : x-_Z18_QSim_rng_sequenceIfEvP4qsimPT_jjj.sm_70.elf.bin



Building Cross-Platform CUDA Applications with CMake
-----------------------------------------------------

* https://developer.nvidia.com/blog/building-cuda-applications-cmake/


Trying CMake CUDA_ARCHITECTURES
----------------------------------

::

    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    project(${name} VERSION 0.1.0 LANGUAGES CXX CUDA)

    #find_package(OpticksCUDA REQUIRED MODULE)
    find_package(CUDAToolkit)


    #CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
    add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "70;75;89")



::

    [  2%] Building CUDA object CMakeFiles/SysRap.dir/SU.cu.o
    /usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler 
           -DCONFIG_Debug -DDEBUG_PIDX -DDEBUG_PIDXYZ -DOPTICKS_OKCONF -DOPTICKS_SYSRAP -DPLOG_LOCAL -DRNG_PHILOX -DSysRap_EXPORTS -DWITH_CHILD -DWITH_CUSTOM4 -DWITH_SLOG -DWITH_STTF 
          --options-file CMakeFiles/SysRap.dir/includes_CUDA.rsp -g 
          -std=c++17 
                     --generate-code=arch=compute_70,code=[compute_70,sm_70] 
                     --generate-code=arch=compute_75,code=[compute_75,sm_75] 
                     --generate-code=arch=compute_89,code=[compute_89,sm_89] 
                     -Xcompiler=-fPIC -MD 
                    -MT CMakeFiles/SysRap.dir/SU.cu.o -MF CMakeFiles/SysRap.dir/SU.cu.o.d -x cu -c /home/blyth/opticks/sysrap/SU.cu -o CMakeFiles/SysRap.dir/SU.cu.o
    [  2%] Linking CXX shared library libSysRap.so



::

    A[blyth@localhost tests]$ cuobjdump -ltext $OPTICKS_PREFIX/lib64/libSysRap.so 
    SASS text section 1 : x-_ZN3cub2-1.sm_70.elf.bin
    SASS text section 2 : x-_ZN3cub2-2.sm_70.elf.bin
    SASS text section 3 : x-_ZN3cub2-3.sm_70.elf.bin
    SASS text section 4 : x-_ZN3cub2-4.sm_70.elf.bin
    SASS text section 5 : x-_ZN3cub2-5.sm_70.elf.bin
    SASS text section 6 : x-_ZN3cub2-6.sm_70.elf.bin
    SASS text section 7 : x-_ZN3cub2-7.sm_70.elf.bin
    SASS text section 8 : x-_ZN3cub2-8.sm_70.elf.bin
    SASS text section 9 : x-_ZN3cub25CUB_200302_700_750_890_NS28DeviceReduceSingleTileKernelINS0_18DeviceReducePolicyIlyN6thrust28THRUST_200302_700_750_890_NS4plusIlEEE9Policy600EPlS9_iS6_llEEvT0_T1_T2_T3_T4_.sm_70.elf.bin
    SASS text section 10 : x-_ZN3cub2-10.sm_70.elf.bin
    SASS text section 11 : x-_ZN3cub2-11.sm_70.elf.bin
    SASS text section 12 : x-_ZN3cub25CUB_200302_700_750_890_NS28DeviceReduceSingleTileKernelINS0_18DeviceReducePolicyIljN6thrust28THRUST_200302_700_750_890_NS4plusIlEEE9Policy600EPlS9_iS6_llEEvT0_T1_T2_T3_T4_.sm_70.elf.bin
    SASS text section 13 : x-_ZN3cub2-13.sm_70.elf.bin
    SASS text section 14 : x-_ZN3cub2-14.sm_70.elf.bin
    SASS text section 15 : x-_ZN3cub25CUB_200302_700_750_890_NS11EmptyKernelIvEEvv.sm_70.elf.bin
    SASS text section 16 : x-_ZN6thru-16.sm_70.elf.bin
    SASS text section 17 : x-_ZN6thru-17.sm_70.elf.bin
    SASS text section 18 : x-_ZN6thru-18.sm_70.elf.bin
    SASS text section 19 : x-_ZN6thrust28THRUST_200302_700_750_890_NS8cuda_cub4core13_kernel_agentINS1_9__copy_if9InitAgentIN3cub25CUB_200302_700_750_890_NS13ScanTileStateIiLb1EEEPiiEES9_mSA_EEvT0_T1_T2_.sm_70.elf.bin
    SASS text section 20 : x-_ZN3cub2-20.sm_75.elf.bin
    SASS text section 21 : x-_ZN3cub2-21.sm_75.elf.bin
    SASS text section 22 : x-_ZN3cub2-22.sm_75.elf.bin
    SASS text section 23 : x-_ZN3cub2-23.sm_75.elf.bin
    SASS text section 24 : x-_ZN3cub2-24.sm_75.elf.bin
    SASS text section 25 : x-_ZN3cub2-25.sm_75.elf.bin
    SASS text section 26 : x-_ZN3cub2-26.sm_75.elf.bin
    SASS text section 27 : x-_ZN3cub2-27.sm_75.elf.bin
    SASS text section 28 : x-_ZN3cub25CUB_200302_700_750_890_NS28DeviceReduceSingleTileKernelINS0_18DeviceReducePolicyIlyN6thrust28THRUST_200302_700_750_890_NS4plusIlEEE9Policy600EPlS9_iS6_llEEvT0_T1_T2_T3_T4_.sm_75.elf.bin
    SASS text section 29 : x-_ZN3cub2-29.sm_75.elf.bin
    SASS text section 30 : x-_ZN3cub2-30.sm_75.elf.bin
    SASS text section 31 : x-_ZN3cub25CUB_200302_700_750_890_NS28DeviceReduceSingleTileKernelINS0_18DeviceReducePolicyIljN6thrust28THRUST_200302_700_750_890_NS4plusIlEEE9Policy600EPlS9_iS6_llEEvT0_T1_T2_T3_T4_.sm_75.elf.bin
    SASS text section 32 : x-_ZN3cub2-32.sm_75.elf.bin
    SASS text section 33 : x-_ZN3cub2-33.sm_75.elf.bin
    SASS text section 34 : x-_ZN3cub25CUB_200302_700_750_890_NS11EmptyKernelIvEEvv.sm_75.elf.bin
    SASS text section 35 : x-_ZN6thru-35.sm_75.elf.bin








nvcc -gencode
---------------

* https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/


Virtual Architecture Feature List
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-feature-list

::

    compute_70 and compute_72                                                         : Volta
    compute_75                                                                        : Turing
    compute_80, compute_86 and compute_87                                             : Ampere
    compute_89                                                                        : Ada
    compute_90, compute_90a                                                           : Hopper
    compute_100, compute_100a compute_101, compute_101a compute_120, compute_120a     : Blackwell


GPU Feature List
~~~~~~~~~~~~~~~~

Like above with "sm_70" ... 

* https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list

GPUs are named sm_xy, where x denotes the GPU generation number, and y the version in that generation





