#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>
#include <cassert>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "scuda.h"    // roundUp
#include "OPTIX_CHECK.h"

#include "Ctx.h"
#include "Binding.h"
#include "PIP.h"


static bool readFile( std::string& str, const char* path )
{
    std::ifstream fp(path);
    if( fp.good() )
    {   
        std::stringstream content ;
        content << fp.rdbuf();
        str = content.str();
        return true;
    }   
    return false;
}

OptixPipelineCompileOptions PIP::CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues ) // static
{
    OptixPipelineCompileOptions pipeline_compile_options = {} ;

    pipeline_compile_options.usesMotionBlur        = false;
    //pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ; 
    // without the OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING  got no intersects

    pipeline_compile_options.numPayloadValues      = numPayloadValues ;   // in optixTrace call
    pipeline_compile_options.numAttributeValues    = numAttributeValues ;
    //pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE; 
    pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    return pipeline_compile_options ;  
}


OptixProgramGroupOptions PIP::CreateProgramGroupOptions() // static
{
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros
    return program_group_options ; 
}

 
PIP::PIP(const char* ptx_path_ ) 
    :
    max_trace_depth(2),
#ifdef WITH_PRD
    num_payload_values(2),     // see 
    num_attribute_values(2),   // see __intersection__is
#else
    num_payload_values(8),     // see 
    num_attribute_values(8),   // see __intersection__is
#endif
    pipeline_compile_options(CreatePipelineOptions(num_payload_values,num_attribute_values)),
    program_group_options(CreateProgramGroupOptions()),
    module(CreateModule(ptx_path_,pipeline_compile_options))
{
    init(); 
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


/**
PIP::init
-----------

Names the programs to form the pipeline  

**/

void PIP::init()
{
    std::cout << "PIP::init " << std::endl ; 
    createRaygenPG("rg");
    createMissPG("ms"); 
    createHitgroupPG("is", "ch", nullptr); 
    linkPipeline(max_trace_depth);
}

/**
PIP::createModule
-------------------

PTX from file is read and compiled into the module

**/

OptixModule PIP::CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options ) // static 
{
    std::string ptx ; 
    readFile(ptx, ptx_path ); 

    std::cout 
        << " ptx_path " << ptx_path << std::endl 
        << " ptx size " << ptx.size() << std::endl 
        ;

    OptixModule module = nullptr ;

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

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

void PIP::linkPipeline(unsigned max_trace_depth)
{
    OptixProgramGroup program_groups[] = { raygen_pg, miss_pg, hitgroup_pg };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = max_trace_depth ;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur = false;

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
    OPTIX_CHECK_LOG( optixUtilAccumulateStackSizes( raygen_pg,   &stackSizes ) ); 
    OPTIX_CHECK_LOG( optixUtilAccumulateStackSizes( miss_pg,     &stackSizes ) ); 
    OPTIX_CHECK_LOG( optixUtilAccumulateStackSizes( hitgroup_pg, &stackSizes ) ); 

    uint32_t max_trace_depth = 1;   // only RG invokes trace, no recursion   
    uint32_t max_cc_depth = 0; 
    uint32_t max_dc_depth = 0; 

    LOG(info) 
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

    LOG(info)
        << "(outputs from optixUtilComputeStackSizes) " 
        << std::endl 
        << " directCallableStackSizeFromTraversal " << directCallableStackSizeFromTraversal
        << std::endl 
        << " directCallableStackSizeFromState " << directCallableStackSizeFromState
        << std::endl 
        << " continuationStackSize " << continuationStackSize
        ;


    unsigned maxTraversableGraphDepth = 2 ;  // Opticks only using IAS->GAS

    LOG(info) 
        << "(further inputs to optixPipelineSetStackSize)"
        << std::endl 
        << " maxTraversableGraphDepth " << maxTraversableGraphDepth
        ;  


    //OPTIX_CHECK_LOG( optixProgramGroupGetStackSize(raygen_pg, OptixStackSizes* stackSizes ) );

    OPTIX_CHECK_LOG( optixPipelineSetStackSize( pipeline,
                                       directCallableStackSizeFromTraversal,
                                       directCallableStackSizeFromState,
                                       continuationStackSize,
                                       maxTraversableGraphDepth ) ) ;
}


