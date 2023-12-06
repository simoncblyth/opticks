try_to_slim_the_kernel
========================

::

    N[blyth@localhost opticks]$ KNOBS=1 ./cxs_min.sh 
    knobs is a function
    knobs () 
    { 
        type $FUNCNAME;
        local exceptionFlags;
        local debugLevel;
        local optLevel;
        exceptionFlags=NONE;
        debugLevel=DEFAULT;
        optLevel=DEFAULT;
        export PIP__CreatePipelineOptions_exceptionFlags=$exceptionFlags;
        export PIP__CreateModule_debugLevel=$debugLevel;
        export PIP__linkPipeline_debugLevel=$debugLevel;
        export PIP__CreateModule_optLevel=$optLevel;
        export Ctx=INFO;
        export PIP=INFO;
        export CSGOptiX=INFO
    }
                    GEOM : J23_1_0_rc3_ok0 
                  LOGDIR : /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL1 
                 BINBASE : /home/blyth/tmp/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest 
                     CVD :  
    CUDA_VISIBLE_DEVICES : 1 
                    SDIR : /data/blyth/junotop/opticks/CSGOptiX 
                    FOLD :  
                     LOG :  
                    NEVT :  
    ./cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2023-12-06 13:02:39.002  002950780
    SLOG::EnvLevel adjusting loglevel by envvar   key CSGOptiX level INFO fallback DEBUG upper_level INFO
    SLOG::EnvLevel adjusting loglevel by envvar   key Ctx level INFO fallback DEBUG upper_level INFO
    SLOG::EnvLevel adjusting loglevel by envvar   key PIP level INFO fallback DEBUG upper_level INFO
    2023-12-06 13:02:40.477 INFO  [365521] [CSGOptiX::Create@280] [ fd.descBase CSGFoundry.descBase 
     CFBase       /home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0
     OriginCFBase /home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0

    2023-12-06 13:02:40.477 INFO  [365521] [CSGOptiX::InitSim@258] [
    2023-12-06 13:02:41.312 INFO  [365521] [CSGOptiX::InitSim@267] ]QSim::desc
     this 0x12e51460 INSTANCE 0x12e51460 QEvent.hh:event 0x12e51510 qsim.h:sim 0x12e2dd70
    2023-12-06 13:02:41.312 INFO  [365521] [CSGOptiX::InitGeo@243] [
    2023-12-06 13:02:41.322 INFO  [365521] [CSGOptiX::InitGeo@245] ]
    2023-12-06 13:02:41.322 INFO  [365521] [CSGOptiX::InitParams@303] [
    2023-12-06 13:02:41.322 INFO  [365521] [CSGOptiX::init@348] [ raygenmode 2 SRG::Name(raygenmode) simulate sim 0x12e51460 event 0x12e51510
    2023-12-06 13:02:41.322 INFO  [365521] [CSGOptiX::init@359]  ptxpath /home/blyth/junotop/ExternalLibs/opticks/head/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
    2023-12-06 13:02:41.322 INFO  [365521] [CSGOptiX::init@360]  geoptxpath -
    2023-12-06 13:02:41.322 INFO  [365521] [CSGOptiX::initCtx@379] [
    2023-12-06 13:02:41.330 INFO  [365521] [Ctx::log_cb@27] [ 4][       KNOBS]: All knobs on default.

    2023-12-06 13:02:41.416 INFO  [365521] [Ctx::log_cb@27] [ 4][  DISK CACHE]: Opened database: "/var/tmp/OptixCache_blyth/optix7cache.db"
    2023-12-06 13:02:41.416 INFO  [365521] [Ctx::log_cb@27] [ 4][  DISK CACHE]:     Cache data size: "33.2 MiB"
    2023-12-06 13:02:41.417 INFO  [365521] [CSGOptiX::initCtx@383] 
    Ctx::desc
    Properties::desc
                          limitMaxTraceDepth :         31
               limitMaxTraversableGraphDepth :         31
                    limitMaxPrimitivesPerGas :  536870912  20000000
                     limitMaxInstancesPerIas :  268435456  10000000
                               rtcoreVersion :          a
                          limitMaxInstanceId :  268435455   fffffff
          limitNumBitsInstanceVisibilityMask :          8
                    limitMaxSbtRecordsPerGas :  268435456  10000000
                           limitMaxSbtOffset :  268435455   fffffff

    2023-12-06 13:02:41.417 INFO  [365521] [CSGOptiX::initCtx@385] ]
    2023-12-06 13:02:41.417 INFO  [365521] [CSGOptiX::initPIP@390] [
    OPT::ExceptionFlags options NONE exceptionFlags 02023-12-06 13:02:41.424 INFO  [365521] [PIP::CreateModule@210] 
     ptx_path /home/blyth/junotop/ExternalLibs/opticks/head/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx
     ptx.size 2347081
     ptx_ok YES

    OPT::OptimizationLevel  option DEFAULT level 0
     option DEFAULT level 0 OPTIX_VERSION 70500
    2023-12-06 13:02:41.424 INFO  [365521] [PIP::CreateModule@231] [PIP::Desc
     PIP__CreateModule_optLevel    DEFAULT
     PIP__CreateModule_debugLevel  DEFAULT
    ]PIP::Desc
    [PIP::Desc_ModuleCompileOptions
     module_compile_options.maxRegisterCount 0 OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0
     module_compile_options.optLevel         0 OPTIX_COMPILE_OPTIMIZATION_DEFAULT
     module_compile_options.debugLevel       0 OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT
    ]PIP::Desc_ModuleCompileOptions

    2023-12-06 13:02:41.429 INFO  [365521] [Ctx::log_cb@27] [ 4][   DISKCACHE]: Cache hit for key: ptx-2347081-key32106f0e2549001ad82fc6b2c774c3f0-sm_75-rtc1-drv515.43.04
    2023-12-06 13:02:41.429 INFO  [365521] [Ctx::log_cb@27] [ 4][COMPILE FEEDBACK]: 
    2023-12-06 13:02:41.429 INFO  [365521] [PIP::init@148] [
     option DEFAULT level 0 OPTIX_VERSION 70500
    2023-12-06 13:02:41.442 INFO  [365521] [Ctx::log_cb@27] [ 4][COMPILE FEEDBACK]: Info: Pipeline has 1 module(s), 4 entry function(s), 3 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 3546 basic block(s) in entry functions, 41354 instruction(s) in entry functions, 7 non-entry function(s), 53 basic block(s) in non-entry functions, 627 instruction(s) in non-entry functions, no debug information








    2023-12-06 13:16:51.105 INFO  [387697] [Ctx::log_cb@27] [ 4][   DISKCACHE]: Inserted module in cache with key: ptx-2334674-key378d14b49ba3cccb7c522f74720d9063-sm_75-rtc1-drv515.43.04
    2023-12-06 13:16:51.105 INFO  [387697] [Ctx::log_cb@27] [ 4][COMPILE FEEDBACK]: Info: Module uses 2 payload values.Info: Module uses 0 attribute values. Pipeline configuration: 2 (default).
    Info: Entry function "__raygen__rg" with semantic type RAYGEN has 1 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 772 basic block(s), 12057 instruction(s)
    Info: Entry function "__miss__ms" with semantic type MISS has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 1 basic block(s), 28 instruction(s)
    Info: Entry function "__closesthit__ch" with semantic type CLOSESTHIT has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 20 basic block(s), 515 instruction(s)
    Info: Entry function "__intersection__is" with semantic type INTERSECTION has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 2735 basic block(s), 28373 instruction(s)
    Info: 7 non-entry function(s) have 53 basic block(s), 627 instruction(s)

    2023-12-06 13:16:51.105 INFO  [387697] [PIP::init@148] [
     option DEFAULT level 0 OPTIX_VERSION 70500
    2023-12-06 13:16:51.116 INFO  [387697] [Ctx::log_cb@27] [ 4][COMPILE FEEDBACK]: Info: Pipeline has 1 module(s), 4 entry function(s), 1 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 3528 basic block(s) in entry functions, 40973 instruction(s) in entry functions, 7 non-entry function(s), 53 basic block(s) in non-entry functions, 627 instruction(s) in non-entry functions, no debug information








    2023-12-06 13:02:41.442 INFO  [365521] [PIP::init@155] ]
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initPIP@395] ]
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initSBT@400] [
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initSBT@405] ]
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initFrameBuffer@410] [
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initFrameBuffer@412] ]
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initCheckSim@418]  sim 0x12e51460 event 0x12e51510
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initStack@430] 
    2023-12-06 13:02:41.442 INFO  [365521] [PIP::configureStack@484] (inputs to optixUtilComputeStackSizes)
     max_trace_depth 1 max_cc_depth 0 max_dc_depth 0
    2023-12-06 13:02:41.442 INFO  [365521] [PIP::configureStack@506] (outputs from optixUtilComputeStackSizes) 
     directCallableStackSizeFromTraversal 0
     directCallableStackSizeFromState 0
     continuationStackSize 1664
    2023-12-06 13:02:41.442 INFO  [365521] [PIP::configureStack@519] (further inputs to optixPipelineSetStackSize)
     maxTraversableGraphDepth 2
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initGeometry@456] [
    2023-12-06 13:02:41.442 INFO  [365521] [CSGOptiX::initGeometry@469] [ sbt.setFoundry 
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::initGeometry@471] ] sbt.setFoundry 
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::setTop@592] [ tspec i0
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::setTop@602] ] tspec i0
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::initGeometry@475] ]
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::initSimulate@518] 
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::init@373] ]
    2023-12-06 13:02:41.510 INFO  [365521] [CSGOptiX::Create@295] ]
    2023-12-06 13:02:41.511 INFO  [365521] [CSGOptiX::prepareSimulateParam@790] 
    2023-12-06 13:02:41.511 INFO  [365521] [CSGOptiX::prepareParam@822] Params::detail
    (values)
    Params::desc
             raygenmode          2
                 handle 139636588412932
                  width       1920
                 height       1080
                  depth          1
             cameratype          0
               origin_x        960
               origin_y        540
                   tmin       0.05
                   tmax      1e+06

    (device pointers)
                   node 0x7effd6657c00
                   plan          0
                   tran          0
                   itra 0x7effd6751400
                 pixels 0x7effd7400000
                  isect          0
                    sim 0x7effd6627600
                    evt 0x7effd6627400

    2023-12-06 13:02:41.512 INFO  [365521] [CSGOptiX::launch@859]  raygenmode 2 SRG::Name(raygenmode) simulate width 1000000 height 1 depth 1
    2023-12-06 13:02:41.906 INFO  [365521] [CSGOptiX::launch@887]  (params.width, params.height, params.depth) ( 1920,1080,1) 0.3939
    2023-12-06 13:02:41.906 INFO  [365521] [QSim::simulate@366]  eventID 0 dt    0.393915

