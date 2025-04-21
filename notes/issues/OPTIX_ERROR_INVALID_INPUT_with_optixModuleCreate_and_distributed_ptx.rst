OPTIX_ERROR_INVALID_INPUT_with_optixModuleCreate_and_distributed_ptx
=======================================================================

Overview
---------

Presumably caused by version differences between build and run machines, 
but which matters

* compute capability 
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



