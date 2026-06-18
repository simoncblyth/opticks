G4CXRenderTest_failing
========================


Overview
-----------

Simplify Params.h and fixing 16 byte alignment makes no difference, still getting
all params values to be zero within kernel.

But find that after the primitive type fix that other scripts are working::

    cxr_min.sh
    cxs_min.sh

What is special with::

     g4cx/tests/G4CXRenderTest.sh

HMM: that has the disclaimer::

  This is not currently expected to produce meaningful renders
  because it lacks all the environment setup of g4cx/gxr.sh



So the lack of params may be actually the case ? YEP: Confirmed
----------------------------------------------------------------

::

    2026-06-17 19:50:39.993 INFO  [3575003] [CSGOptiX::prelaunch_sanity_check@1159]  eye( 0 0 0)
    2026-06-17 19:50:39.993 INFO  [3575003] [CSGOptiX::prelaunch_sanity_check@1167]  params_helper.detail
    Params_Helper::detail
    (values)
    Params_Helper::desc
             raygenmode          0
                 handle 140305923833860
                  width       1920
                 height       1080
                  depth          1
             cameratype          0
             traceyflip          0
             rendertype          0
               origin_x        960
               origin_y        540
                   tmin          0
                   tmax          0
                    eye(  0.000000  0.000000  0.000000  0.000000)
                      U(  0.000000  0.000000  0.000000  0.000000)
                      V(  0.000000  0.000000  0.000000  0.000000)
                      W(  0.000000  0.000000  0.000000  0.000000)
                  WNORM(  0.000000  0.000000  0.000000  0.000000)

    (device pointers)
                   node 0x10017a00000
                   plan          0
                   tran          0
                   itra 0x10017839200
                 pixels 0x7f9b85400000
                  isect          0
                    sim          0
                    evt          0

    //CSGOptiX7.cu:render idx(10,10,0) dim(1920,1080,1) params.cameratype:0 params.origin_x:960 params.U(  0.000,  0.000,  0.000,  0.000) params.V(  0.000,  0.000,  0.000,  0.000) params._pad(0,0,0) 
    2026-06-17 19:50:40.244 INFO  [3575003] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [INVALID_RAY] invalid ray encountered
        An invalid ray was passed to optixTrace.
            origin    (0 0 0)
            direction (nan nan nan)
            tmin      0
            tmax      0
            time      0

        /home/blyth/opticks/CSGOptiX/CSGOptiX7.cu:135:9
        Launch index (260,48,0)

    Error launching work to RTX
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_params, sizeof_Params , &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1127)

    ./GXTestRunner.sh: line 51: 3575003 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXRenderTest
    g4cx/tests/G4CXRenderTest.sh : run error




Refs on optix9.1
-------------------

* https://forums.developer.nvidia.com/t/optix-9-1-release/354119

* https://raytracing-docs.nvidia.com/optix9/api/group__optix__host__api__launches.html


pipelineParamsSize number of bytes are copied from the device memory pointed to
by pipelineParams before launch. It is an error if pipelineParamsSize is
greater than the size of the variable declared in modules and identified by
OptixPipelineCompileOptions::pipelineLaunchParamsVariableName. If the launch
params variable was optimized out or not found in the modules linked to the
pipeline then the pipelineParams and pipelineParamsSize parameters are ignored.


Issue - Render optixLaunch failure
-------------------------------------

::

    SLOW: tests taking longer that 15.0 seconds


    FAILS:  1   / 221   :  Tue Jun 16 16:34:07 2026  :  GEOM J26_1_1_opticks_Debug  
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                                 ***Failed                      4.25   



OPTIX_Exception::

    [lo] A[blyth@localhost opticks]$  g4cx/tests/G4CXRenderTest.sh
             BASH_SOURCE : g4cx/tests/G4CXRenderTest.sh 
                    GEOM : J26_1_1_opticks_Debug 
                     bin : G4CXRenderTest 
    ./GXTestRunner.sh - use externaly set GEOM CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /home/blyth/opticks/g4cx/tests
                    GEOM : J26_1_1_opticks_Debug
    J26_1_1_opticks_Debug_GDMLPathFromGEOM : 
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXRenderTest
                    ARGS : 
    2026-06-17 10:49:38.917 INFO  [3425099] [main@27] [ cu first 
    2026-06-17 10:49:39.031 INFO  [3425099] [main@29] ] cu first 
    2026-06-17 10:49:39.031 INFO  [3425099] [main@35] [ SetGeometry 
    2026-06-17 10:49:45.989 INFO  [3425099] [main@37] ] SetGeometry 
    2026-06-17 10:49:45.989 INFO  [3425099] [main@39] [ gx->render 
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1103)

    ./GXTestRunner.sh: line 51: 3425099 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXRenderTest
    g4cx/tests/G4CXRenderTest.sh : run error
    [lo] A[blyth@localhost opticks]$ 



Enable debug in g4cx/tests/G4CXRenderTest.sh::

    optix_kernel_debug()
    {
        type $FUNCNAME
        export PIP__CreateModule_optLevel=LEVEL_0
        export PIP__CreateModule_debugLevel=FULL
        export PIP__CreatePipelineOptions_exceptionFlags="TRACE_DEPTH|STACK_OVERFLOW"
        export Ctx=INFO  # set Ctx::LEVEL to see optix kernel callback logging
    }
    [ -n "$OPTIX_KERNEL_DEBUG" ] && optix_kernel_debug



::

    [lo] A[blyth@localhost opticks]$ OPTIX_KERNEL_DEBUG=1 g4cx/tests/G4CXRenderTest.sh
    optix_kernel_debug is a function
    optix_kernel_debug () 
    { 
        type $FUNCNAME;
        export PIP__CreateModule_optLevel=LEVEL_0;
        export PIP__CreateModule_debugLevel=FULL;
        export PIP__CreatePipelineOptions_exceptionFlags="TRACE_DEPTH|STACK_OVERFLOW";
        export Ctx=INFO
    }
             BASH_SOURCE : g4cx/tests/G4CXRenderTest.sh 
                    GEOM : J26_1_1_opticks_Debug 
                     bin : G4CXRenderTest 
    ./GXTestRunner.sh - use externaly set GEOM CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /home/blyth/opticks/g4cx/tests
                    GEOM : J26_1_1_opticks_Debug
    J26_1_1_opticks_Debug_GDMLPathFromGEOM : 
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXRenderTest
                    ARGS : 
    SLOG::EnvLevel adjusting loglevel by envvar   key Ctx level INFO fallback DEBUG upper_level INFO
    2026-06-17 10:50:40.441 INFO  [3425220] [main@27] [ cu first 
    2026-06-17 10:50:40.552 INFO  [3425220] [main@29] ] cu first 
    2026-06-17 10:50:40.552 INFO  [3425220] [main@35] [ SetGeometry 
    2026-06-17 10:50:41.510 INFO  [3425220] [Ctx::log_cb@50] [ 4][       KNOBS]: All OptiX knobs on default.

    2026-06-17 10:50:41.570 INFO  [3425220] [Ctx::log_cb@50] [ 4][   DISKCACHE]: Opened database: "/var/tmp/OptixCache_blyth/optix7cache.db"
    2026-06-17 10:50:41.570 INFO  [3425220] [Ctx::log_cb@50] [ 4][   DISKCACHE]:     Cache data size: "328.2 MiB"
    2026-06-17 10:50:41.608 INFO  [3425220] [Ctx::log_cb@50] [ 4][   DISKCACHE]: Cache hit for key: ptx-2931016-keyaaccd5d5bc95e0695d95f459a7c3717a-sm_89-rtc1-drv610.43.02

    2026-06-17 10:50:41.608 INFO  [3425220] [Ctx::log_cb@50] [ 4][    COMPILER]: 
    2026-06-17 10:50:41.609 INFO  [3425220] [Ctx::log_cb@50] [ 4][   DISKCACHE]: Cache hit for key: ptx-93-keye63e129053b6e512ee17efcc367bcba6-sm_89-rtc1-drv610.43.02

    2026-06-17 10:50:41.609 INFO  [3425220] [Ctx::log_cb@50] [ 4][    COMPILER]: 
    2026-06-17 10:50:41.618 INFO  [3425220] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Pipeline statistics
        module(s)                            :     1
        entry function(s)                    :     4
        trace call(s)                        :     7
        continuation callable call(s)        :     0
        direct callable call(s)              :     0
        basic block(s) in entry functions    :  1926
        instruction(s) in entry functions    : 27200
        non-entry function(s)                :     3
        basic block(s) in non-entry functions:   144
        instruction(s) in non-entry functions:  2059
        debug information                    :    no

    2026-06-17 10:50:41.680 INFO  [3425220] [main@37] ] SetGeometry 
    2026-06-17 10:50:41.680 INFO  [3425220] [main@39] [ gx->render 
    //CSGOptiX7.cu:render idx(10,10,0) dim(1920,1080,1) 
    2026-06-17 10:50:41.910 INFO  [3425220] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [INVALID_RAY] invalid ray encountered
        An invalid ray was passed to optixTrace.
            origin    (0 0 0)
            direction (nan nan nan)
            tmin      0
            tmax      0
            time      0

        /home/blyth/opticks/CSGOptiX/CSGOptiX7.cu:133:9
        Launch index (292,88,0)

    Error launching work to RTX
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1103)

    ./GXTestRunner.sh: line 51: 3425220 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXRenderTest
    g4cx/tests/G4CXRenderTest.sh : run error
    [lo] A[blyth@localhost opticks]$ 




Adding param dumping, seems all zero::

    2026-06-17 11:10:49.596 INFO  [3428164] [main@37] ] SetGeometry 
    2026-06-17 11:10:49.596 INFO  [3428164] [main@39] [ gx->render 
    //CSGOptiX7.cu:render idx(10,10,0) dim(1920,1080,1) params.cameratype:0 params.U(  0.000,  0.000,  0.000) params.V(  0.000,  0.000,  0.000)  
    2026-06-17 11:10:49.848 INFO  [3428164] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [INVALID_RAY] invalid ray encountered
        An invalid ray was passed to optixTrace.
            origin    (0 0 0)
            direction (nan nan nan)
            tmin      0
            tmax      0
            time      0



Looks like params not being uploaded::

    [lo] A[blyth@localhost opticks]$ OPTIX_KERNEL_DEBUG=1 BP=CSGOptiX::prepareParam g4cx/tests/G4CXRenderTest.sh dbg

    (gdb) bt
    #0  CSGOptiX::prepareParam (this=0x1543d0d0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1015
    #1  0x00007ffff48a0542 in CSGOptiX::launch (this=0x1543d0d0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1057
    #2  0x00007ffff48a117f in CSGOptiX::render_launch (this=0x1543d0d0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1154
    #3  0x00007ffff48a17cd in CSGOptiX::render (this=0x1543d0d0, stem_=0x0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1265
    #4  0x00007ffff7ed0823 in G4CXOpticks::render (this=0xf61b50) at /home/blyth/opticks/g4cx/G4CXOpticks.cc:539
    #5  0x0000000000403bbc in main (argc=1, argv=0x7fffffffb878) at /home/blyth/opticks/g4cx/tests/G4CXRenderTest.cc:40
    (gdb) f 0
    #0  CSGOptiX::prepareParam (this=0x1543d0d0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1015
    1015	    const glm::tvec4<double>& ce = sglm->fr.ce ;
    (gdb) list
    1010	
    1011	**/
    1012	
    1013	void CSGOptiX::prepareParam()
    1014	{
    1015	    const glm::tvec4<double>& ce = sglm->fr.ce ;
    1016	
    1017	    params->setCenterExtent(ce.x, ce.y, ce.z, ce.w);
    1018	    switch(raygenmode)
    1019	    {
    (gdb) 
    1020	        case SRG_RENDER   : prepareParamRender()   ; break ;
    1021	        case SRG_SIMTRACE : prepareParamSimulate() ; break ;
    1022	        case SRG_SIMULATE : prepareParamSimulate() ; break ;
    1023	    }
    1024	
    1025	    params->upload();
    1026	    LOG_IF(level, !flight) << params->detail();
    1027	}
    1028	
    1029	
    (gdb) b Params::upload
    Breakpoint 2 at 0x7ffff48876db: file /home/blyth/opticks/CSGOptiX/Params.cc, line 203.
    (gdb) 





::

    [lo] A[blyth@localhost opticks]$ l /var/tmp/OptixCache_blyth/
    total 431748
         4 drwxrwxrwt. 16 root  root       4096 Jun 17 15:13 ..
     67100 -rw-r--r--.  1 blyth blyth  68706912 Jun 17 15:13 optix7cache.db-wal
        32 -rw-r--r--.  1 blyth blyth     32768 Jun 17 15:13 optix7cache.db-shm
    364612 -rw-r--r--.  1 blyth blyth 373358592 Jun 17 11:01 optix7cache.db
         0 drwxr-xr--.  2 blyth blyth        80 Jun 17 10:11 .
    [lo] A[blyth@localhost opticks]$ date
    Wed Jun 17 03:16:19 PM CST 2026
    [lo] A[blyth@localhost opticks]$ 










Simplify Params.h and fixing 16 byte alignment makes no difference, still getting
all params values to be zero::


    [lo] A[blyth@localhost opticks]$ g4cx/tests/G4CXRenderTest.sh
    optix_kernel_debug is a function
    optix_kernel_debug () 
    { 
        type $FUNCNAME;
        export PIP__CreateModule_optLevel=LEVEL_0;
        export PIP__CreateModule_debugLevel=FULL;
        export PIP__CreatePipelineOptions_exceptionFlags="TRACE_DEPTH|STACK_OVERFLOW";
        export Ctx=INFO;
        export OPTIX_CACHE_MAXSIZE=0
    }
             BASH_SOURCE : g4cx/tests/G4CXRenderTest.sh 
                    GEOM : J26_1_1_opticks_Debug 
                     bin : G4CXRenderTest 
    ./GXTestRunner.sh - use externaly set GEOM CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /home/blyth/opticks/g4cx/tests
                    GEOM : J26_1_1_opticks_Debug
    J26_1_1_opticks_Debug_GDMLPathFromGEOM : 
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXRenderTest
                    ARGS : 
    SLOG::EnvLevel adjusting loglevel by envvar   key Ctx level INFO fallback DEBUG upper_level INFO
    2026-06-17 17:08:03.603 INFO  [3515801] [main@27] [ cu first 
    2026-06-17 17:08:03.721 INFO  [3515801] [main@29] ] cu first 
    2026-06-17 17:08:03.721 INFO  [3515801] [main@35] [ SetGeometry 
    2026-06-17 17:08:04.678 INFO  [3515801] [Ctx::log_cb@50] [ 4][       KNOBS]: All OptiX knobs on default.

    2026-06-17 17:08:04.720 INFO  [3515801] [Ctx::log_cb@50] [ 4][   DISKCACHE]: OPTIX_CACHE_MAXSIZE is set to 0. Disabling the OptiX disk cache. The cache contents will not be changed.
    2026-06-17 17:08:05.210 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Warning: Requested debug level "OPTIX_COMPILE_DEBUG_LEVEL_FULL", but input module does not include full debug information.
    Info: Pipeline parameter "params" size is 320 bytes

    2026-06-17 17:08:06.227 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __closesthit__ch__ptID_1__0xa7b69ceb8fddedcd
        register count                  :   128
        direct stack size (bytes)       :    56
        direct spills (bytes)           :     8
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 17:08:06.227 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __intersection__is__ptID_1__0xa7b69ceb8fddedcd
        register count                  :   128
        direct stack size (bytes)       :  2040
        direct spills (bytes)           :  1336
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 17:08:06.227 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __miss__ms__ptID_1__0xa7b69ceb8fddedcd
        register count                  :   123
        direct stack size (bytes)       :    40
        direct spills (bytes)           :     0
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 17:08:06.227 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
        payload values        :          2
        attribute values      :          0
    Info: Properties for entry function "__miss__ms"
        semantic type                :                   MISS
        trace call(s)                :                      0
        continuation callable call(s):                      0
        basic block(s)               :                      1
        instruction(s)               :                     46
    Info: Properties for entry function "__closesthit__ch"
        semantic type                :             CLOSESTHIT
        trace call(s)                :                      0
        continuation callable call(s):                      0
        basic block(s)               :                     51
        instruction(s)               :                   1209
    Info: Properties for entry function "__intersection__is"
        semantic type                :           INTERSECTION
        trace call(s)                :                      0
        continuation callable call(s):                      0
        basic block(s)               :                    704
        instruction(s)               :                   8367
    Info: Compiled Module Summary
        non-entry function(s):     3
        basic block(s)       :   144
        instruction(s)       :  2059

    2026-06-17 17:08:12.771 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __raygen__rg__0xa7b69ceb8fddedcd
        register count                  :   128
        direct stack size (bytes)       :  2392
        direct spills (bytes)           :  6196
        continuation stack size (bytes) :  2208
        continuation spills (bytes)     :  3336

    2026-06-17 17:08:12.774 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
        payload values        :          2
        attribute values      :          0
    Info: Properties for entry function "__raygen__rg"
        semantic type                :                 RAYGEN
        trace call(s)                :                      7
        continuation callable call(s):                      0
        basic block(s)               :                   1170
        instruction(s)               :                  17672
    Info: Compiled Module Summary
        non-entry function(s):     0
        basic block(s)       :     0
        instruction(s)       :     0

    2026-06-17 17:08:12.798 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __raygen__rg_dummy__0xa7b69ceb8fddedcd
        register count                  :   108
        direct stack size (bytes)       :    48
        direct spills (bytes)           :     0
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 17:08:12.798 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
        payload values        :          0
        attribute values      :          0
    Info: Properties for entry function "__raygen__rg_dummy"
        semantic type                :                 RAYGEN
        trace call(s)                :                      0
        continuation callable call(s):                      0
        basic block(s)               :                      1
        instruction(s)               :                     19
    Info: Compiled Module Summary
        non-entry function(s):     0
        basic block(s)       :     0
        instruction(s)       :     0

    2026-06-17 17:08:12.803 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: 
    2026-06-17 17:08:12.829 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __exception__default__0xc86833901b415d4b
        register count                  :    29
        direct stack size (bytes)       :     0
        direct spills (bytes)           :     0
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 17:08:12.829 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
        payload values        :          0
        attribute values      :          0
    Info: Properties for entry function "__exception__default"
        semantic type                :              EXCEPTION
        trace call(s)                :                      0
        continuation callable call(s):                      0
        basic block(s)               :                     52
        instruction(s)               :                    277
    Info: Compiled Module Summary
        non-entry function(s):     0
        basic block(s)       :     0
        instruction(s)       :     0

    2026-06-17 17:08:12.836 INFO  [3515801] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Pipeline statistics
        module(s)                            :     1
        entry function(s)                    :     4
        trace call(s)                        :     7
        continuation callable call(s)        :     0
        direct callable call(s)              :     0
        basic block(s) in entry functions    :  1926
        instruction(s) in entry functions    : 27294
        non-entry function(s)                :     3
        basic block(s) in non-entry functions:   144
        instruction(s) in non-entry functions:  2059
        debug information                    :    no

    Params_Helper::device_alloc d_param address is 0x10017bc8400
    2026-06-17 17:08:12.901 INFO  [3515801] [main@37] ] SetGeometry 
    2026-06-17 17:08:12.901 INFO  [3515801] [main@39] [ gx->render 
    Params_Helper::upload d_param address is 0x10017bc8400 sizeof_Params 320
    LAUNCH: d_param address is  0: 0x10017bc8400 1: 1099909858304 sizeof_Params 320
    Params_Helper::upload d_param address is 0x10017bc8400 sizeof_Params 320
    //CSGOptiX7.cu:render idx(10,10,0) dim(1920,1080,1) params.cameratype:0 params.U(  0.000,  0.000,  0.000,  0.000) params.V(  0.000,  0.000,  0.000,  0.000) params._pad(0,0,0) 
    //CSGOptiX7.cu:render idx(10,10,0) cameratype:0 params.U(  0.000,  0.000,  0.000) params.V(  0.000,  0.000,  0.000) direction(    nan,    nan,    nan) 
    2026-06-17 17:08:13.149 INFO  [3515801] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [INVALID_RAY] invalid ray encountered
        An invalid ray was passed to optixTrace.
            origin    (0 0 0)
            direction (nan nan nan)
            tmin      0
            tmax      0
            time      0

        /home/blyth/opticks/CSGOptiX/CSGOptiX7.cu:135:9
        Launch index (28,24,0)

    Error launching work to RTX
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof_Params , &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1124)

    ./GXTestRunner.sh: line 51: 3515801 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXRenderTest
    g4cx/tests/G4CXRenderTest.sh : run error
    [lo] A[blyth@localhost opticks]$ 







