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
#include "OPT.h"
#include "PIP.h"
#include "SLOG.hh"

const plog::Severity PIP::LEVEL = SLOG::EnvLevel("PIP", "DEBUG"); 




bool PIP::OptiXVersionIsSupported()  // static
{
    bool ok = false ; 
#if OPTIX_VERSION == 70000 || OPTIX_VERSION == 70500 || OPTIX_VERSION == 70600 
    ok = true ; 
#elif OPTIX_VERSION >= 80000
    ok = true ;
#endif
    return ok ; 
}


const char* PIP::CreatePipelineOptions_exceptionFlags  = ssys::getenvvar("PIP__CreatePipelineOptions_exceptionFlags", "STACK_OVERFLOW" ); 

/**
PIP::CreatePipelineOptions
----------------------------

traversableGraphFlags
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS : got no intersects with this
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING  : works 
 
usesPrimitiveTypeFlags
    from optix7-;optix7-types
    Setting to zero corresponds to enabling OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM and OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE.

    Changed form unset 0 to OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM removing OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE

**/


OptixPipelineCompileOptions PIP::CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues ) // static
{
    unsigned traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ;

    OptixPipelineCompileOptions pipeline_compile_options = {} ;
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = traversableGraphFlags ; 
    pipeline_compile_options.numPayloadValues      = numPayloadValues ;   // in optixTrace call
    pipeline_compile_options.numAttributeValues    = numAttributeValues ;
    pipeline_compile_options.exceptionFlags        = OPT::ExceptionFlags( CreatePipelineOptions_exceptionFlags )  ;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM ;  

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
       << OPT::Desc_ExceptionFlags( pipeline_compile_options.exceptionFlags )
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

const int PIP::MAX_TRACE_DEPTH = ssys::getenvint("PIP__max_trace_depth", 1 ) ;   // was 2 

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


PIP::~PIP()
{
    destroy(); 
}

/**
PIP::init
-----------

Names the programs to form the pipeline  

**/

void PIP::init()
{
    LOG(LEVEL)  << "[" ; 

    createRaygenPG();
    createMissPG(); 
    createHitgroupPG(); 
    linkPipeline(max_trace_depth);

    LOG(LEVEL)  << "]" ; 
}


void PIP::destroy()
{
    destroyPipeline(); 
    destroyRaygenPG();
    destroyMissPG();
    destroyHitgroupPG();
    destroyModule(); 
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
       << " " << OPT::OptimizationLevel_( module_compile_options.optLevel )
       << std::endl 
       << " module_compile_options.debugLevel       " << module_compile_options.debugLevel  
       << " " << OPT::DebugLevel_( module_compile_options.debugLevel )
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

    LOG_IF(fatal, !ptx_ok)
        << std::endl 
        << " ptx_path " << ptx_path 
        << std::endl 
        << " ptx.size " << ptx.size() 
        << std::endl 
        << " ptx_ok " << ( ptx_ok ? "YES" : "NO " ) 
        << std::endl 
        ; 

    LOG(LEVEL)
        << std::endl 
        << " ptx_path " << ptx_path 
        << std::endl 
        << " ptx.size " << ptx.size() 
        << std::endl 
        << " ptx_ok " << ( ptx_ok ? "YES" : "NO " ) 
        << std::endl 
        ; 
    assert(ptx_ok); 

    int maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ; 
    OptixCompileOptimizationLevel optLevel = OPT::OptimizationLevel(CreateModule_optLevel) ; 
    OptixCompileDebugLevel      debugLevel = OPT::DebugLevel(CreateModule_debugLevel) ; 

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

void PIP::destroyModule()
{
    OPTIX_CHECK( optixModuleDestroy( module ) );
}


/**
PIP::createRaygenPG
---------------------

Creates member raygen_pg

**/



void PIP::createRaygenPG()
{
    bool DUMMY = ssys::getenvbool("PIP__createRaygenPG_DUMMY"); 
    LOG(LEVEL) << " DUMMY " << ( DUMMY ? "YES" : "NO " ) ; 


    OptixProgramGroupDesc desc    = {}; 
    desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module            = module;
    desc.raygen.entryFunctionName = DUMMY ? RG_DUMMY : RG ;

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

void PIP::destroyRaygenPG()
{
    OPTIX_CHECK( optixProgramGroupDestroy( raygen_pg ) );
}



/**
PIP::createMissPG
---------------------

Creates member miss_pg

**/

void PIP::createMissPG()
{
    OptixProgramGroupDesc desc  = {};
    desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module            = module;
    desc.miss.entryFunctionName = MS ;

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

void PIP::destroyMissPG()
{
    OPTIX_CHECK( optixProgramGroupDestroy( miss_pg ) );
}




/**
PIP::createHitgroupPG
---------------------

**/

void PIP::createHitgroupPG()
{
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    desc.hitgroup.moduleIS            = module ;
    desc.hitgroup.entryFunctionNameIS =  IS ;

    desc.hitgroup.moduleCH            = module  ; 
    desc.hitgroup.entryFunctionNameCH = CH ;

    desc.hitgroup.moduleAH            = nullptr ;
    desc.hitgroup.entryFunctionNameAH = nullptr ; 

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

void PIP::destroyHitgroupPG()
{
    OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_pg ) );
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
       << " " << OPT::DebugLevel_(pipeline_link_options.debugLevel ) 
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

    OptixCompileDebugLevel debugLevel = OPT::DebugLevel(linkPipeline_debugLevel)  ; 
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

void PIP::destroyPipeline()
{
    OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
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

    // see optix7-;optix7-host : it states that IAS->GAS needs to be two  
    unsigned maxTraversableGraphDepth = 2 ; 

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


