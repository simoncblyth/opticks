#!/bin/bash 
usage(){ cat << EOU

cxr_min_debug.sh 
=================

Run cxr_min.sh with env modified to enable debug logging, eg::

    2025-04-21 10:26:33.868 INFO  [330236] [Ctx::log_cb@43] [ 4][   DISKCACHE]: Cache miss for key: ptx-2057151-keyff735237e4c1f641ab0cb523124ecf91-sm_75-rtc1-drv515.43.04
    2025-04-21 10:26:34.493 INFO  [330236] [Ctx::log_cb@43] [ 4][COMPILE FEEDBACK]: Info: Pipeline parameter "params" size is 288 bytes

    2025-04-21 10:26:37.889 INFO  [330236] [Ctx::log_cb@43] [ 4][   DISKCACHE]: Inserted module in cache with key: ptx-2057151-keyff735237e4c1f641ab0cb523124ecf91-sm_75-rtc1-drv515.43.04
    2025-04-21 10:26:37.889 INFO  [330236] [Ctx::log_cb@43] [ 4][COMPILE FEEDBACK]: Info: Module uses 2 payload values.Info: Module uses 0 attribute values. Pipeline configuration: 2 (default).
    Info: Entry function "__raygen__rg_dummy" with semantic type RAYGEN has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 1 basic block(s), 1 instruction(s)
    Info: Entry function "__raygen__rg" with semantic type RAYGEN has 3 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 1127 basic block(s), 16440 instruction(s)
    Info: Entry function "__miss__ms" with semantic type MISS has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 1 basic block(s), 28 instruction(s)
    Info: Entry function "__closesthit__ch" with semantic type CLOSESTHIT has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 43 basic block(s), 1143 instruction(s)
    Info: Entry function "__intersection__is" with semantic type INTERSECTION has 0 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 1383 basic block(s), 15768 instruction(s)
    Info: 7 non-entry function(s) have 53 basic block(s), 627 instruction(s)

    2025-04-21 10:26:37.889 INFO  [330236] [PIP::init@171] [
    2025-04-21 10:26:37.889 INFO  [330236] [PIP::createRaygenPG@331]  DUMMY NO 
    2025-04-21 10:26:37.899 INFO  [330236] [Ctx::log_cb@43] [ 4][COMPILE FEEDBACK]: Info: Pipeline has 1 module(s), 4 entry function(s), 3 trace call(s), 0 continuation callable call(s), 0 direct callable call(s), 2554 basic block(s) in entry functions, 33379 instruction(s) in entry functions, 7 non-entry function(s), 53 basic block(s) in non-entry functions, 627 instruction(s) in non-entry functions, no debug information

    2025-04-21 10:26:37.899 INFO  [330236] [PIP::init@178] ]
    2025-04-21 10:26:37.899 INFO  [330236] [PIP::configureStack@568] (inputs to optixUtilComputeStackSizes)


Notice the cache key incorporates both sm_75 and drv515.43.04::

   ptx-2057151-keyff735237e4c1f641ab0cb523124ecf91-sm_75-rtc1-drv515.43.04


EOU
}


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


cxr_min.sh 


