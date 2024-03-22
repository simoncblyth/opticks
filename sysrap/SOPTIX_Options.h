#pragma once
/**
SOPTIX_Options.h
=================

moduleCompileOptions.maxRegisterCount
    OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT:0 for no limit 
    TODO: expt changing this


**/

#include "ssys.h"
#include "SOPTIX_OPT.h"

struct SOPTIX_Options
{
    static constexpr const char* SOPTIX_Options_optLevel = "SOPTIX_Options_optLevel" ; 
    static constexpr const char* SOPTIX_Options_debugLevel = "SOPTIX_Options_debugLevel" ; 
    static constexpr const char* SOPTIX_Options_exceptionFlags = "SOPTIX_Options_exceptionFlags" ; 

    const char* _optLevel ; 
    const char* _debugLevel ; 
    const char* _exceptionFlags ; 

    unsigned _numPayloadValues ;   
    unsigned _numAttributeValues ;   

    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};

    std::string desc() const ;
 
    SOPTIX_Options() ; 
    void init();

    void init_ModuleCompile();
    static std::string Desc_ModuleCompileOptions(const OptixModuleCompileOptions& module_compile_options );

    void init_PipelineCompile();
    static std::string Desc_PipelineCompileOptions(const OptixPipelineCompileOptions& pipeline_compile_options );
}; 

inline std::string SOPTIX_Options::desc() const
{
    std::stringstream ss ;
    ss << "[SOPTIX_Options::desc" << std::endl ;
    ss << " _optLevel " << _optLevel << std::endl ;
    ss << " _debugLevel " << _debugLevel << std::endl ;
    ss << " _exceptionFlags " << _exceptionFlags << std::endl ;
    ss << Desc_ModuleCompileOptions(moduleCompileOptions) ;  
    ss << Desc_PipelineCompileOptions(pipelineCompileOptions) ;  
    ss << "]SOPTIX_Options::desc" << std::endl ;
    std::string str = ss.str() ; 
    return str ; 
}

inline SOPTIX_Options::SOPTIX_Options()
    :
    _optLevel(   ssys::getenvvar(SOPTIX_Options_optLevel, "DEFAULT" ) ), 
    _debugLevel( ssys::getenvvar(SOPTIX_Options_debugLevel, "DEFAULT" ) ),
    _exceptionFlags( ssys::getenvvar(SOPTIX_Options_exceptionFlags, "STACK_OVERFLOW" ) ),
    _numPayloadValues(2),  
    _numAttributeValues(2)  
{
    init();
}

inline void SOPTIX_Options::init()
{
    init_ModuleCompile();
    init_PipelineCompile();
}

/**
SOPTIX_Options::init_ModuleCompile
------------------------------------

moduleCompileOptions.numPayloadTypes
   Must be zero if OptixPipelineCompileOptions::numPayloadValues is not zero

**/

inline void SOPTIX_Options::init_ModuleCompile()
{
    OptixCompileOptimizationLevel optLevel = SOPTIX_OPT::OptimizationLevel(_optLevel) ;
    OptixCompileDebugLevel debugLevel = SOPTIX_OPT::DebugLevel(_debugLevel) ; 

    const OptixModuleCompileBoundValueEntry* boundValues = nullptr ; 
    unsigned numBoundValues = 0 ; 
    unsigned numPayloadTypes = 0 ;     
    OptixPayloadType* payloadTypes = nullptr ;    

    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ; 
    moduleCompileOptions.optLevel = optLevel ;
    moduleCompileOptions.debugLevel = debugLevel ;
    moduleCompileOptions.boundValues = boundValues ;
    moduleCompileOptions.numBoundValues = numBoundValues ; 
    moduleCompileOptions.numPayloadTypes = numPayloadTypes ;
    moduleCompileOptions.payloadTypes = payloadTypes ; 
}

inline std::string SOPTIX_Options::Desc_ModuleCompileOptions(const OptixModuleCompileOptions& module_compile_options )
{
    std::stringstream ss ; 
    ss  
       << "[SOPTIX_Options::Desc_ModuleCompileOptions" << std::endl 
       << " module_compile_options.maxRegisterCount " << module_compile_options.maxRegisterCount  
       << " OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT " << OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
       << std::endl 
       << " module_compile_options.optLevel         " << module_compile_options.optLevel  
       << " " << SOPTIX_OPT::OptimizationLevel_( module_compile_options.optLevel )
       << std::endl 
       << " module_compile_options.debugLevel       " << module_compile_options.debugLevel  
       << " " << SOPTIX_OPT::DebugLevel_( module_compile_options.debugLevel )
       << std::endl 
       << "]SOPTIX_Options::Desc_ModuleCompileOptions" 
       << std::endl 
       ;   
    std::string str = ss.str() ; 
    return str ; 
}

inline void SOPTIX_Options::init_PipelineCompile()
{
    int usesMotionBlur = 0 ; 
    OptixTraversableGraphFlags traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ;
    unsigned numPayloadValues = _numPayloadValues ; // in optixTrace call 
    unsigned numAttributeValues = _numAttributeValues ; 
    unsigned exceptionFlags = SOPTIX_OPT::ExceptionFlags( _exceptionFlags ) ;
    const char* pipelineLaunchParamsVariableName = "params";
    unsigned usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ; 

    pipelineCompileOptions.usesMotionBlur        = usesMotionBlur ;
    pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags ; 
    pipelineCompileOptions.numPayloadValues      = numPayloadValues ;   // in optixTrace call
    pipelineCompileOptions.numAttributeValues    = numAttributeValues ; 
    pipelineCompileOptions.exceptionFlags        = exceptionFlags ; 
    pipelineCompileOptions.pipelineLaunchParamsVariableName = pipelineLaunchParamsVariableName ; 
    pipelineCompileOptions.usesPrimitiveTypeFlags = usesPrimitiveTypeFlags ;
}

inline std::string SOPTIX_Options::Desc_PipelineCompileOptions(const OptixPipelineCompileOptions& pipeline_compile_options )
{
    std::stringstream ss ;
    ss
       << "[SOPTIX_Options::Desc_PipelineCompileOptions" << std::endl
       << " pipeline_compile_options.numPayloadValues   " << pipeline_compile_options.numPayloadValues  << std::endl
       << " pipeline_compile_options.numAttributeValues " << pipeline_compile_options.numAttributeValues << std::endl
       << " pipeline_compile_options.exceptionFlags     " << pipeline_compile_options.exceptionFlags
       << SOPTIX_OPT::Desc_ExceptionFlags( pipeline_compile_options.exceptionFlags )
       << std::endl
       << "]SOPTIX_Options::Desc_PipelineCompileOptions" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}

