cxr_min_also_failing_with_optix910_with_UNSUPPORTED_PRIMITIVE_TYPE
===================================================================


Overview
---------

Seems that the optix9.1 CUDA 13.1  KMD Version: 610.43.02 (Driver?) is stricter.


Issue : cxr_min.sh gives black screen and then exception
----------------------------------------------------------

::


    [lo] A[blyth@localhost CSGOptiX]$ cxr_min.sh
                                 GEOM_METHOD : local sourcing of ~/.opticks/GEOM/GEOM.sh 
                                        GEOM : J26_1_1_opticks_Debug 
                                     NOXGEOM :  
                     External_CFBaseFromGEOM : _CFBaseFromGEOM 
                             _CFBaseFromGEOM :  
                       BASH_SOURCE : /home/blyth/.opticks/GEOM/ELV.sh
                              GEOM : J26_1_1_opticks_Debug
                          elv_name : skip_big
                          ELV_NAME : skip_big


    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof_Params , &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1127)

    /data1/blyth/local/opticks_Debug/bin/cxr_min.sh: line 370: 3539423 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxr_min.sh run error



With more debug, crucial error message::

    2026-06-17 18:13:01.474 INFO  [3539226] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [UNSUPPORTED_PRIMITIVE_TYPE] unsupported primitive type encountered
        optixTrace encountered a primitive type not supported by the pipeline. Supported primitive types are specified in OptixPipelineCompileOptions::usesPrimitiveTypeFlags.
        Launch index (80,22,0)

    Error launching work to RTX
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof_Params , &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1127)

 



FIX : Adding CUSTOM and TRIANGLE, regains cxr_min.sh operation
----------------------------------------------------------------

::

     64 OptixPipelineCompileOptions PIP::CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues ) // static
     65 {
     66     unsigned traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ;
     67     //unsigned usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM ; // <<< UNTIL 2026/06/17
     68     unsigned usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM|OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ;
     69 
     70     OptixPipelineCompileOptions pipeline_compile_options = {} ;
     71     pipeline_compile_options.usesMotionBlur        = false;
     72     pipeline_compile_options.traversableGraphFlags = traversableGraphFlags ;
     73     pipeline_compile_options.numPayloadValues      = numPayloadValues ;   // in optixTrace call
     74     pipeline_compile_options.numAttributeValues    = numAttributeValues ;
     75     pipeline_compile_options.exceptionFlags        = OPT::ExceptionFlags( CreatePipelineOptions_exceptionFlags )  ;
     76     pipeline_compile_options.pipelineLaunchParamsVariableName = pipelineLaunchParamsVariableName ;
     77     pipeline_compile_options.usesPrimitiveTypeFlags = usesPrimitiveTypeFlags ;
     78 
     79     return pipeline_compile_options ;
     80 }




Investigate issue by enabling kernel debug via envvars
---------------------------------------------------------

Reveals the cause to be UNSUPPORTED_PRIMITIVE_TYPE::

    2026-06-17 18:13:01.474 INFO  [3539226] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [UNSUPPORTED_PRIMITIVE_TYPE] unsupported primitive type encountered
        optixTrace encountered a primitive type not supported by the pipeline. Supported primitive types are specified in OptixPipelineCompileOptions::usesPrimitiveTypeFlags.
        Launch index (80,22,0)



Pump up the volume::

    [lo] A[blyth@localhost CSGOptiX]$ OPTIX_KERNEL_DEBUG=1 cxr_min.sh
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
                                 GEOM_METHOD : local sourcing of ~/.opticks/GEOM/GEOM.sh 
                                        GEOM : J26_1_1_opticks_Debug 
                                     NOXGEOM :  
                     External_CFBaseFromGEOM : _CFBaseFromGEOM 
                             _CFBaseFromGEOM :  
                       BASH_SOURCE : /home/blyth/.opticks/GEOM/ELV.sh
                              GEOM : J26_1_1_opticks_Debug
                          elv_name : skip_big
                          ELV_NAME : skip_big
                               elv : t:sWorld,sBottomRock_withLowerRect,sTopRock,sExpHallPlusER2,sExpRockWithMiddle,sDomeRockBox,sTopRock_domeAir,sTopRock_dome_plusUpper,sDomeRockBox,sAirTT,PoolCoversub,sElecRoom1,sElecRoom2,sTyvek_shell,sAirGap,sOuterWaterPool,sPoolLining,sTarget_T,sAcrylic_T,sOuterWaterInCD_T,sOuterReflectorInCD_T,sInnerWater_T,sInnerReflectorInCD_T,sBar_0,sBar_1,sPanelTape,sPanel,sPlane_0,sPlane_1,sWall_0,sWall_1,sWall_2,HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual
                               ELV : t:sWorld,sBottomRock_withLowerRect,sTopRock,sExpHallPlusER2,sExpRockWithMiddle,sDomeRockBox,sTopRock_domeAir,sTopRock_dome_plusUpper,sDomeRockBox,sAirTT,PoolCoversub,sElecRoom1,sElecRoom2,sTyvek_shell,sAirGap,sOuterWaterPool,sPoolLining,sTarget_T,sAcrylic_T,sOuterWaterInCD_T,sOuterReflectorInCD_T,sInnerWater_T,sInnerReflectorInCD_T,sBar_0,sBar_1,sPanelTape,sPanel,sPlane_0,sPlane_1,sWall_0,sWall_1,sWall_2,HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual
                        elv_branch : J26_1_1_opticks_Debug/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/skip_big:437
                        ELV_BRANCH : J26_1_1_opticks_Debug/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/skip_big:437
    vue_logging is a function
    vue_logging () 
    { 
        : ~/.opticks/GEOM/VUE.sh;
        type $FUNCNAME;
        export CSGFoundry__getFrame_VERBOSE=1;
        export CSGFoundry__getFrameE_VERBOSE=1
    }
    /data1/blyth/local/opticks_Debug/bin/cxr_min.sh : USING EXTERNALLY SETUP GEOMETRY ENVIRONMENT : EG FROM OJ DISTRIBUTION
                     bin : CSGOptiXRenderInteractiveTest 
               which_bin : /data1/blyth/local/opticks_Debug/lib/CSGOptiXRenderInteractiveTest 
                    GEOM : J26_1_1_opticks_Debug 
                     MOI : EXTENT:10000 
                MOI_NOTE : 176 
                     EMM :  
                     ELV : t:sWorld,sBottomRock_withLowerRect,sTopRock,sExpHallPlusER2,sExpRockWithMiddle,sDomeRockBox,sTopRock_domeAir,sTopRock_dome_plusUpper,sDomeRockBox,sAirTT,PoolCoversub,sElecRoom1,sElecRoom2,sTyvek_shell,sAirGap,sOuterWaterPool,sPoolLining,sTarget_T,sAcrylic_T,sOuterWaterInCD_T,sOuterReflectorInCD_T,sInnerWater_T,sInnerReflectorInCD_T,sBar_0,sBar_1,sPanelTape,sPanel,sPlane_0,sPlane_1,sWall_0,sWall_1,sWall_2,HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual 
                ELV_NAME : skip_big 
              ELV_BRANCH : J26_1_1_opticks_Debug/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/skip_big:437 
                    TMIN : 0.5 
                     VUE : Y 
                     EYE : 0,1,0 
                    LOOK : 0,0,0 
                      UP : 0,0,1 
                    ZOOM : 1 
                  LOGDIR : /data1/blyth/tmp/GEOM/J26_1_1_opticks_Debug/CSGOptiXRenderInteractiveTest 
                    BASE : /data1/blyth/tmp/GEOM/J26_1_1_opticks_Debug/CSGOptiXRenderInteractiveTest 
                    PBAS : /data1/blyth/tmp/ 
              NAMEPREFIX : cxr_min__eye_0,1,0__zoom_1__tmin_0.5_ 
            OPTICKS_HASH :  
                 TOPLINE : ESCALE=extent EYE=0,1,0 TMIN=0.5 MOI=EXTENT:10000 ZOOM=1 CAM=perspective cxr_min.sh  
                 BOTLINE : Wed Jun 17 06:12:51 PM CST 2026 
    CUDA_VISIBLE_DEVICES : 0 
                   AFOLD :  
      AFOLD_RECORD_SLICE :  
                   BFOLD :  
      BFOLD_RECORD_SLICE :  
                    _CUR : GEOM/J26_1_1_opticks_Debug/cxr_min/ 
    SLOG::EnvLevel adjusting loglevel by envvar   key Ctx level INFO fallback DEBUG upper_level INFO
    SRecord::LoadArray FAILED : DUE TO MISSING ENVVAR
     _fold [$AFOLD]
     path [AFOLD/record.npy]
     looks_unresolved YES
    SRecord::LoadArray FAILED : DUE TO MISSING ENVVAR
     _fold [$BFOLD]
     path [BFOLD/record.npy]
     looks_unresolved YES
    SGLM::initView VIEW [-] load_interpolated_view NO  interpolated_view.brief -
    2026-06-17 18:12:52.820 INFO  [3539226] [Ctx::log_cb@50] [ 4][       KNOBS]: All OptiX knobs on default.

    2026-06-17 18:12:52.860 INFO  [3539226] [Ctx::log_cb@50] [ 4][   DISKCACHE]: OPTIX_CACHE_MAXSIZE is set to 0. Disabling the OptiX disk cache. The cache contents will not be changed.
    2026-06-17 18:12:53.334 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Warning: Requested debug level "OPTIX_COMPILE_DEBUG_LEVEL_FULL", but input module does not include full debug information.
    Info: Pipeline parameter "params" size is 320 bytes

    2026-06-17 18:12:54.334 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __closesthit__ch__ptID_1__0xa7b69ceb8fddedcd
        register count                  :   128
        direct stack size (bytes)       :    56
        direct spills (bytes)           :     8
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 18:12:54.334 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __intersection__is__ptID_1__0xa7b69ceb8fddedcd
        register count                  :   128
        direct stack size (bytes)       :  2040
        direct spills (bytes)           :  1336
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 18:12:54.334 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __miss__ms__ptID_1__0xa7b69ceb8fddedcd
        register count                  :   123
        direct stack size (bytes)       :    40
        direct spills (bytes)           :     0
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 18:12:54.334 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
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

    2026-06-17 18:13:00.988 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __raygen__rg__0xa7b69ceb8fddedcd
        register count                  :   128
        direct stack size (bytes)       :  2392
        direct spills (bytes)           :  6196
        continuation stack size (bytes) :  2208
        continuation spills (bytes)     :  3336

    2026-06-17 18:13:00.991 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
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

    2026-06-17 18:13:01.014 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __raygen__rg_dummy__0xa7b69ceb8fddedcd
        register count                  :   108
        direct stack size (bytes)       :    48
        direct spills (bytes)           :     0
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 18:13:01.014 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
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

    2026-06-17 18:13:01.019 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: 
    2026-06-17 18:13:01.044 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Function properties for __exception__default__0xc86833901b415d4b
        register count                  :    29
        direct stack size (bytes)       :     0
        direct spills (bytes)           :     0
        continuation stack size (bytes) :     0
        continuation spills (bytes)     :     0

    2026-06-17 18:13:01.044 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Module Statistics
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

    2026-06-17 18:13:01.052 INFO  [3539226] [Ctx::log_cb@50] [ 4][    COMPILER]: Info: Pipeline statistics
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

    Params_Helper::device_alloc d_param address is 0x100179f5400
    Params_Helper::upload d_param address is 0x100179f5400 sizeof_Params 320 sizeof_params 320
    LAUNCH: d_param address is  0: 0x100179f5400 1: 1099907945472 sizeof_Params 320 width 2560 height 1440 depth 1
    Params_Helper::upload d_param address is 0x100179f5400 sizeof_Params 320 sizeof_params 320
    //CSGOptiX7.cu:render idx(10,10,0) dim(2560,1440,1) params.cameratype:0 params.U(-8888.889,  0.000,  0.000,  0.000) params.V(  0.000,  0.000,5000.000,  0.000) params._pad(0,0,0) 
    //CSGOptiX7.cu:render idx(10,10,0) cameratype:0 params.U(-8888.889,  0.000,  0.000) params.V(  0.000,  0.000,5000.000) direction(  0.782, -0.444, -0.437) 
    2026-06-17 18:13:01.474 INFO  [3539226] [Ctx::log_cb@50] [ 2][       ERROR]: Error syncing stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    [UNSUPPORTED_PRIMITIVE_TYPE] unsupported primitive type encountered
        optixTrace encountered a primitive type not supported by the pipeline. Supported primitive types are specified in OptixPipelineCompileOptions::usesPrimitiveTypeFlags.
        Launch index (80,22,0)

    Error launching work to RTX
    Error recording resource event on user stream (CUDA error string: unspecified launch failure, CUDA error code: 719)
    terminate called after throwing an instance of 'OPTIX_Exception'
      what():  OPTIX_ERROR_LAUNCH_FAILURE: Optix call 'optixLaunch( pip->pipeline, (CUstream)stream, d_param, sizeof_Params , &(sbt->sbt), width, height, depth )' failed: /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1127)

    /data1/blyth/local/opticks_Debug/bin/cxr_min.sh: line 370: 3539226 Aborted                 (core dumped) $bin
    /data1/blyth/local/opticks_Debug/bin/cxr_min.sh run error
    [lo] A[blyth@localhost CSGOptiX]$ 









