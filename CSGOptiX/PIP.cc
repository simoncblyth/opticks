#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>
#include <cassert>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <cuda_runtime.h>
#include "scuda.h"    // roundUp
#include "sstr.h"
#include "ssys.h"
#include "spath.h"

#include "OPTIX_CHECK.h"

#include "Ctx.h"
#include "Binding.h"
#include "PIP.h"

#include "SLOG.hh"

const plog::Severity PIP::LEVEL = SLOG::EnvLevel("PIP", "DEBUG"); 




bool PIP::OptiXVersionIsSupported()  // static
{
    bool ok = false ; 
#if OPTIX_VERSION == 70000 || OPTIX_VERSION == 70500 || OPTIX_VERSION == 70600
    ok = true ; 
#endif
    return ok ; 
}

/**
PIP::DebugLevel
-----------------

https://forums.developer.nvidia.com/t/gpu-program-optimization-questions/195238/2

droettger, Nov 2021::

    The new OPTIX_COMPILE_DEBUG_LEVEL_MODERATE is documented to have an impact on
    performance.  You should use OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL which keeps only
    the line information for profiling and OPTIX_COMPILE_DEBUG_LEVEL_NONE to remove
    even that.  Never profile compute kernels build as debug! That will completely
    change the code structure and does not represent the fully optimized code.


See optix7-;optix7-types

**/


OptixCompileDebugLevel PIP::DebugLevel(const char* option)  // static
{
    OptixCompileDebugLevel level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
#if OPTIX_VERSION == 70000
    if(     strcmp(option, "NONE") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
    else if(strcmp(option, "LINEINFO") == 0 ) level = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO ; 
    else if(strcmp(option, "FULL") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_FULL ; 
    else if(strcmp(option, "DEFAULT") == 0 )  level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
#elif OPTIX_VERSION == 70500 || OPTIX_VERSION == 70600
    if(     strcmp(option, "DEFAULT") == 0 )  level = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT ; 
    else if(strcmp(option, "NONE") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_NONE ; 
    else if(strcmp(option, "MINIMAL") == 0 )  level = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL ; 
    else if(strcmp(option, "MODERATE") == 0 ) level = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE ; 
    else if(strcmp(option, "FULL") == 0 )     level = OPTIX_COMPILE_DEBUG_LEVEL_FULL ; 
#else
    LOG(fatal) << " NOT RECOGNIZED " << " option " << option  << " level " << level  
               << " OPTIX_VERSION " << OPTIX_VERSION ; 
    assert(0);   
#endif
    LOG(LEVEL) << " option " << option << " level " << level << " OPTIX_VERSION " << OPTIX_VERSION ;  
    return level ; 
}
const char * PIP::DebugLevel_( OptixCompileDebugLevel debugLevel )
{
    const char* s = nullptr ; 
    switch(debugLevel)
    {  
#if OPTIX_VERSION == 70000
        case OPTIX_COMPILE_DEBUG_LEVEL_NONE:     s = OPTIX_COMPILE_DEBUG_LEVEL_NONE_     ; break ; 
        case OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO: s = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO_ ; break ;
        case OPTIX_COMPILE_DEBUG_LEVEL_FULL:     s = OPTIX_COMPILE_DEBUG_LEVEL_FULL_     ; break ;
#elif OPTIX_VERSION == 70500 || OPTIX_VERSION == 70600
        case OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT:  s = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT_  ; break ; 
        case OPTIX_COMPILE_DEBUG_LEVEL_NONE:     s = OPTIX_COMPILE_DEBUG_LEVEL_NONE_     ; break ; 
        case OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:  s = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL_  ; break ;
        case OPTIX_COMPILE_DEBUG_LEVEL_MODERATE: s = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE_ ; break ;
        case OPTIX_COMPILE_DEBUG_LEVEL_FULL:     s = OPTIX_COMPILE_DEBUG_LEVEL_FULL_     ; break ;
#endif
    }

    if( s == nullptr )
    {
        LOG(fatal) 
            << " IS NOT RECOGNIZED  "
            << " debugLevel " << debugLevel 
            << " OPTIX_VERSION " << OPTIX_VERSION 
            ; 
    }
    return s ;    
}

OptixCompileOptimizationLevel PIP::OptimizationLevel(const char* option) // static 
{
    OptixCompileOptimizationLevel level ; 
    if(      strcmp(option, "LEVEL_0") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0  ; 
    else if( strcmp(option, "LEVEL_1") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1  ; 
    else if( strcmp(option, "LEVEL_2") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2  ; 
    else if( strcmp(option, "LEVEL_3") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3  ; 
    else if( strcmp(option, "DEFAULT") == 0 )  level = OPTIX_COMPILE_OPTIMIZATION_DEFAULT  ; 
    else 
    {
        LOG(fatal) 
            << " IS NOT RECOGNIZED  "
            << " option " << option 
            << " level " << level 
            << " OPTIX_VERSION " << OPTIX_VERSION 
            ; 
        assert(0) ; 
    }
 
    LOG(LEVEL) << " option " << option << " level " << level ;  
    return level ; 
}
const char* PIP::OptimizationLevel_( OptixCompileOptimizationLevel optLevel )
{
    const char* s = nullptr ; 
    switch(optLevel)
    {
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_0: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_1: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_1_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_2: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_2_ ; break ; 
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_3: s = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3_ ; break ; 
        default:                                                                         ; break ; 
    }

    if( s == nullptr )
    {
        LOG(fatal) 
            << " IS NOT RECOGNIZED  "
            << " optLevel " << optLevel 
            << " OPTIX_VERSION " << OPTIX_VERSION 
            ; 
        assert(0) ; 
    }
    return s ; 
} 
OptixExceptionFlags PIP::ExceptionFlags_(const char* opt)
{
    OptixExceptionFlags flag = OPTIX_EXCEPTION_FLAG_NONE ; 
    if(      strcmp(opt, "NONE") == 0 )          flag = OPTIX_EXCEPTION_FLAG_NONE ;  
    else if( strcmp(opt, "STACK_OVERFLOW") == 0) flag = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW ; 
    else if( strcmp(opt, "TRACE_DEPTH") == 0)    flag = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH  ; 
    else if( strcmp(opt, "USER") == 0)           flag = OPTIX_EXCEPTION_FLAG_USER  ; 
    else if( strcmp(opt, "DEBUG") == 0)          flag = OPTIX_EXCEPTION_FLAG_DEBUG  ; 
    else 
    {
        LOG(fatal) 
            << " IS NOT RECOGNIZED  "
            << " opt " << opt
            << " flag " << flag
            << " OPTIX_VERSION " << OPTIX_VERSION 
            ; 
        assert(0) ; 
    }
    return flag ; 
}
const char* PIP::ExceptionFlags__(OptixExceptionFlags excFlag)
{
    const char* s = nullptr ; 
    switch(excFlag)
    {
        case OPTIX_EXCEPTION_FLAG_NONE:           s = OPTIX_EXCEPTION_FLAG_NONE_            ; break ; 
        case OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW: s = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW_  ; break ;
        case OPTIX_EXCEPTION_FLAG_TRACE_DEPTH:    s = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH_     ; break ; 
        case OPTIX_EXCEPTION_FLAG_USER:           s = OPTIX_EXCEPTION_FLAG_USER_            ; break ; 
        case OPTIX_EXCEPTION_FLAG_DEBUG:          s = OPTIX_EXCEPTION_FLAG_DEBUG_           ; break ;      
    }
    return s ; 
}

unsigned PIP::ExceptionFlags(const char* options)
{
    std::vector<std::string> opts ; 
    sstr::Split( options, '|', opts );  

    unsigned exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE ; 
    for(unsigned i=0 ; i < opts.size() ; i++)
    {
        const std::string& opt = opts[i] ; 
        exceptionFlags |= ExceptionFlags_(opt.c_str()); 
    }
    LOG(LEVEL) << " options " << options << " exceptionFlags " << exceptionFlags ; 
    return exceptionFlags ;  
}

std::string PIP::Desc_ExceptionFlags( unsigned flags )
{
    std::stringstream ss ; 
    if( flags & OPTIX_EXCEPTION_FLAG_NONE )           ss << OPTIX_EXCEPTION_FLAG_NONE_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW ) ss << OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_TRACE_DEPTH )    ss << OPTIX_EXCEPTION_FLAG_TRACE_DEPTH_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_USER )           ss << OPTIX_EXCEPTION_FLAG_USER_ << " " ; 
    if( flags & OPTIX_EXCEPTION_FLAG_DEBUG )          ss << OPTIX_EXCEPTION_FLAG_DEBUG_ << " " ; 
    std::string str = ss.str() ; 
    return str ; 
}




const char* PIP::CreatePipelineOptions_exceptionFlags  = ssys::getenvvar("PIP__CreatePipelineOptions_exceptionFlags", "STACK_OVERFLOW" ); 

OptixPipelineCompileOptions PIP::CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues ) // static
{
    //unsigned traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ; 
    unsigned traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ;  // without this get no intersects

    OptixPipelineCompileOptions pipeline_compile_options = {} ;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = traversableGraphFlags ; 
    pipeline_compile_options.numPayloadValues      = numPayloadValues ;   // in optixTrace call
    pipeline_compile_options.numAttributeValues    = numAttributeValues ;
    pipeline_compile_options.exceptionFlags        = ExceptionFlags( CreatePipelineOptions_exceptionFlags )  ;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    return pipeline_compile_options ;  
}

std::string PIP::Desc_PipelineCompileOptions(const OptixPipelineCompileOptions& pipeline_compile_options )
{
    std::stringstream ss ; 
    ss 
       << "[PIP::Desc_PipelineCompileOptions" << std::endl 
       << " pipeline_compile_options.numPayloadValues   " << pipeline_compile_options.numPayloadValues  << std::endl 
       << " pipeline_compile_options.numAttributeValues " << pipeline_compile_options.numAttributeValues << std::endl
       << " pipeline_compile_options.exceptionFlags     " << pipeline_compile_options.exceptionFlags  
       << Desc_ExceptionFlags( pipeline_compile_options.exceptionFlags )
       << std::endl 
       << "]PIP::Desc_PipelineCompileOptions" << std::endl 
       ;
    std::string str = ss.str() ; 
    return str ; 
}






OptixProgramGroupOptions PIP::CreateProgramGroupOptions() // static
{
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros
    return program_group_options ; 
}


const char* PIP::desc() const
{
    std::stringstream ss ; 
    ss << "PIP " 
       << " td:" << max_trace_depth 
       << " pv:" << num_payload_values 
       << " av:" << num_attribute_values 
#ifdef WITH_PRD
       << " WITH_PRD " 
#else
       << " NOT_WITH_PRD " 
#endif    
       << " " 
       ; 

   std::string s = ss.str(); 
   return strdup(s.c_str()); 
}

const int PIP::MAX_TRACE_DEPTH = ssys::getenvint("PIP_max_trace_depth", 1 ) ;   // was 2 

/**
PIP::PIP
---------

PTX read from *ptx_path_* is used to CreateModule

* num_payload_values and num_attribute_values MUST MATCH payload and attribute slots 
  used in the PTX, see CSGOptiX7.cu  

**/
PIP::PIP(const char* ptx_path_, const Properties* properties_ ) 
    :
    max_trace_depth(MAX_TRACE_DEPTH),
#ifdef WITH_PRD
    num_payload_values(2),     // see trace
    num_attribute_values(2),   // see __intersection__is
#else
    num_payload_values(8),     // see trace and setPayload 
    num_attribute_values(6),   // see __intersection__is
#endif
    properties(properties_),
    pipeline_compile_options(CreatePipelineOptions(num_payload_values,num_attribute_values)),
    program_group_options(CreateProgramGroupOptions()),
    module(CreateModule(ptx_path_,pipeline_compile_options))
{
    init(); 
}

/**
PIP::init
-----------

Names the programs to form the pipeline  

**/

void PIP::init()
{
    LOG(LEVEL)  << "[" ; 

    createRaygenPG("rg");
    createMissPG("ms"); 
    createHitgroupPG("is", "ch", nullptr); 
    linkPipeline(max_trace_depth);

    LOG(LEVEL)  << "]" ; 
}

/**
PIP::CreateModule
-------------------

PTX from file is read and compiled into the module

**/

const char* PIP::CreateModule_optLevel   = ssys::getenvvar("PIP__CreateModule_optLevel", "DEFAULT" ) ; 
const char* PIP::CreateModule_debugLevel = ssys::getenvvar("PIP__CreateModule_debugLevel", "DEFAULT" ) ; 

std::string PIP::Desc()
{
    std::stringstream ss ; 
    ss 
       << "[PIP::Desc" << std::endl 
       << " PIP__CreateModule_optLevel    " << CreateModule_optLevel << std::endl 
       << " PIP__CreateModule_debugLevel  " << CreateModule_debugLevel << std::endl 
       << "]PIP::Desc" << std::endl 
       ;

    std::string str = ss.str() ; 
    return str ; 
}


std::string PIP::Desc_ModuleCompileOptions(const OptixModuleCompileOptions& module_compile_options )
{
    std::stringstream ss ; 
    ss 
       << "[PIP::Desc_ModuleCompileOptions" << std::endl 
       << " module_compile_options.maxRegisterCount " << module_compile_options.maxRegisterCount  
       << " OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT " << OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
       << std::endl 
       << " module_compile_options.optLevel         " << module_compile_options.optLevel  
       << " " << OptimizationLevel_( module_compile_options.optLevel )
       << std::endl 
       << " module_compile_options.debugLevel       " << module_compile_options.debugLevel  
       << " " << DebugLevel_( module_compile_options.debugLevel )
       << std::endl 
       << "]PIP::Desc_ModuleCompileOptions" 
       << std::endl 
       ;
    std::string str = ss.str() ; 
    return str ; 
}




OptixModule PIP::CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options ) // static 
{
    std::string ptx ; 
    bool ptx_ok = spath::Read(ptx, ptx_path ); 

    LOG(LEVEL)
        << " ptx_path " << ptx_path << std::endl 
        << " ptx.size " << ptx.size() << std::endl 
        << " ptx_pl " << ( ptx_ok ? "YES" : "NO " ) << std::endl 
        ; 
    assert(ptx_ok); 

    int maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ; 
    OptixCompileOptimizationLevel optLevel = OptimizationLevel(CreateModule_optLevel) ; 
    OptixCompileDebugLevel      debugLevel = DebugLevel(CreateModule_debugLevel) ; 

    OptixModule module = nullptr ;
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = maxRegisterCount ;
    module_compile_options.optLevel             = optLevel ; 
    module_compile_options.debugLevel           = debugLevel ;

    LOG(LEVEL) 
        << Desc()
        << Desc_ModuleCompileOptions( module_compile_options ) 
        ; 


    size_t sizeof_log = 0 ; 
    char log[2048]; // For error reporting from OptiX creation functions

    OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                Ctx::context,
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
                ) );

    return module ; 
}

/**
PIP::createRaygenPG
---------------------

Creates member raygen_pg

**/

void PIP::createRaygenPG(const char* rg)
{
    std::string rg_ = "__raygen__" ; 
    rg_ += rg ;  

    OptixProgramGroupDesc desc    = {}; 
    desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module            = module;
    desc.raygen.entryFunctionName = rg_.c_str() ;

    size_t sizeof_log = 0 ; 
    char log[2048]; 
    unsigned num_program_groups = 1 ; 

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                Ctx::context,
                &desc,
                num_program_groups,
                &program_group_options,
                log,
                &sizeof_log,
                &raygen_pg
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ; 
    assert( sizeof_log == 0); 
}

/**
PIP::createMissPG
---------------------

Creates member miss_pg

**/

void PIP::createMissPG(const char* ms)
{
    std::string ms_ = "__miss__" ; 
    ms_ += ms ;  

    OptixProgramGroupDesc desc  = {};
    desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module            = module;
    desc.miss.entryFunctionName = ms_.c_str() ;

    size_t sizeof_log = 0 ; 
    char log[2048]; 
    unsigned num_program_groups = 1 ; 

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                Ctx::context,
                &desc,
                num_program_groups,
                &program_group_options,
                log,
                &sizeof_log,
                &miss_pg
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ; 
    assert( sizeof_log == 0); 
}

/**
PIP::createHitgroupPG
---------------------

Creates member hitgroup_pg

**/

void PIP::createHitgroupPG(const char* is, const char* ch, const char* ah )
{
    std::string is_ = "__intersection__" ; 
    std::string ch_ = "__closesthit__" ; 
    std::string ah_ = "__anyhit__" ; 

    if(is) is_ += is ;  
    if(ch) ch_ += ch ;  
    if(ah) ah_ += ah ;  

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    if(is)  
    { 
        desc.hitgroup.moduleIS            = module ;
        desc.hitgroup.entryFunctionNameIS =  is_.c_str() ;
    }
    if(ch)
    {
        desc.hitgroup.moduleCH            = module  ; 
        desc.hitgroup.entryFunctionNameCH = ch_.c_str() ;
    }
    if(ah)
    {
        desc.hitgroup.moduleAH            = module ;
        desc.hitgroup.entryFunctionNameAH = ah_.c_str();
    }

    size_t sizeof_log = 0 ; 
    char log[2048]; 
    unsigned num_program_groups = 1 ; 


    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                Ctx::context,
                &desc,
                num_program_groups,
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_pg
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ; 
    assert( sizeof_log == 0); 
}



/**
PIP::linkPipeline
-------------------

Create pipeline from the program_groups

**/


const char* PIP::linkPipeline_debugLevel = ssys::getenvvar("PIP__linkPipeline_debugLevel", "DEFAULT" ) ; 


std::string PIP::Desc_PipelineLinkOptions(const OptixPipelineLinkOptions& pipeline_link_options )
{
    std::stringstream ss ; 
    ss 
       << "[PIP::Desc_PipelineLinkOptions" << std::endl 
       << " pipeline_link_options.maxTraceDepth " << pipeline_link_options.maxTraceDepth << std::endl 
       << " pipeline_link_options.debugLevel    " << pipeline_link_options.debugLevel 
       << " " << DebugLevel_(pipeline_link_options.debugLevel ) 
       << std::endl
       << " PIP__linkPipeline_debugLevel " << linkPipeline_debugLevel 
       << std::endl
       << "]PIP::Desc_PipelineLinkOptions" << std::endl 
       ;
    std::string str = ss.str() ; 
    return str ; 
}


void PIP::linkPipeline(unsigned max_trace_depth)
{
    OptixProgramGroup program_groups[] = { raygen_pg, miss_pg, hitgroup_pg };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = max_trace_depth ;

#if OPTIX_VERSION == 70000
    pipeline_link_options.overrideUsesMotionBlur = false;
#elif OPTIX_VERSION == 70500
#endif

    OptixCompileDebugLevel debugLevel = DebugLevel(linkPipeline_debugLevel)  ; 
    pipeline_link_options.debugLevel = debugLevel;

    size_t sizeof_log = 0 ; 
    char log[2048]; 

    OPTIX_CHECK_LOG( optixPipelineCreate(
                Ctx::context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                log,
                &sizeof_log,
                &pipeline
                ) );

    if(sizeof_log > 0) std::cout << log << std::endl ; 
    assert( sizeof_log == 0); 
}



/**
PIP::configureStack
------------------------

https://on-demand.gputechconf.com/siggraph/2019/pdf/sig915-optix-performance-tools-tricks.pdf

https://sibr.gitlabpages.inria.fr/docs/0.9.6/OptixRaycaster_8cpp_source.html

/Developer/OptiX_700/include/optix_stack_size.h
 
/Developer/OptiX_700/SDK/optixPathTracer/optixPathTracer.cpp

In optix_stack_size.h:
   OptixStackSizes
   optixUtilAccumulateStackSizes()
   optixUtilComputeStackSizes()
   optixPipelineSetStackSize()
These analyze your shaders. Source code included!
Detailed control over stack size See optixPathTracer for example
 

**/
void PIP::configureStack()
{
    // following OptiX_700/SDK/optixPathTracer/optixPathTracer.cpp

    OptixStackSizes stackSizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( raygen_pg,   &stackSizes ) ); 
    OPTIX_CHECK( optixUtilAccumulateStackSizes( miss_pg,     &stackSizes ) ); 
    OPTIX_CHECK( optixUtilAccumulateStackSizes( hitgroup_pg, &stackSizes ) ); 

    uint32_t max_trace_depth = 1;   // only RG invokes trace, no recursion   
    uint32_t max_cc_depth = 0; 
    uint32_t max_dc_depth = 0; 

    LOG(LEVEL) 
        << "(inputs to optixUtilComputeStackSizes)" 
        << std::endl 
        << " max_trace_depth " << max_trace_depth
        << " max_cc_depth " << max_cc_depth
        << " max_dc_depth " << max_dc_depth
        ;

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;

    OPTIX_CHECK( optixUtilComputeStackSizes(
                &stackSizes,
                max_trace_depth,
                max_cc_depth,
                max_dc_depth,
                &directCallableStackSizeFromTraversal,
                &directCallableStackSizeFromState,
                &continuationStackSize
                ) ); 

    LOG(LEVEL)
        << "(outputs from optixUtilComputeStackSizes) " 
        << std::endl 
        << " directCallableStackSizeFromTraversal " << directCallableStackSizeFromTraversal
        << std::endl 
        << " directCallableStackSizeFromState " << directCallableStackSizeFromState
        << std::endl 
        << " continuationStackSize " << continuationStackSize
        ;


    unsigned maxTraversableGraphDepth = 2 ;  // Opticks only using IAS->GAS

    LOG(LEVEL) 
        << "(further inputs to optixPipelineSetStackSize)"
        << std::endl 
        << " maxTraversableGraphDepth " << maxTraversableGraphDepth
        ;  


    //OPTIX_CHECK_LOG( optixProgramGroupGetStackSize(raygen_pg, OptixStackSizes* stackSizes ) );

    OPTIX_CHECK( optixPipelineSetStackSize( pipeline,
                                       directCallableStackSizeFromTraversal,
                                       directCallableStackSizeFromState,
                                       continuationStackSize,
                                       maxTraversableGraphDepth ) ) ;
}


