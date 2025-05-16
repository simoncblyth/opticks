#pragma once
/**
SOPTIX_Options.h : module and pipeline compile/link options
==============================================================

moduleCompileOptions.maxRegisterCount
    OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT:0 for no limit 
    TODO: expt changing this


**/

#include "ssys.h"
#include "SOPTIX_OPT.h"

struct SOPTIX_Options
{
    static constexpr const char* _LEVEL = "SOPTIX_Options__LEVEL" ;
    static int Level(); 

    static constexpr const char* SOPTIX_Options_optLevel = "SOPTIX_Options_optLevel" ; 
    static constexpr const char* SOPTIX_Options_debugLevel = "SOPTIX_Options_debugLevel" ; 
    static constexpr const char* SOPTIX_Options_exceptionFlags = "SOPTIX_Options_exceptionFlags" ; 
    static constexpr const char* SOPTIX_Options_link_debugLevel = "SOPTIX_Options_link_debugLevel" ; 

    const char* _optLevel ; 
    const char* _debugLevel ; 
    const char* _exceptionFlags ; 
    const char* _link_debugLevel ; 


    unsigned _numPayloadValues ;   
    unsigned _numAttributeValues ; 
  
    static constexpr const char* SOPTIX_Options_maxTraceDepth = "SOPTIX_Options_maxTraceDepth" ; 
    unsigned _maxTraceDepth ; 

    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixProgramGroupOptions programGroupOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    std::string desc() const ;
 
    SOPTIX_Options() ; 
    void init();

    void init_moduleCompileOptions();
    static std::string Desc_moduleCompileOptions(const OptixModuleCompileOptions& module_compile_options );

    void init_pipelineCompileOptions();
    static std::string Desc_pipelineCompileOptions(const OptixPipelineCompileOptions& pipeline_compile_options );

    void init_pipelineLinkOptions();
    static std::string Desc_pipelineLinkOptions(const OptixPipelineLinkOptions& pipeline_link_options );

}; 

inline int SOPTIX_Options::Level()
{
    return ssys::getenvint(_LEVEL, 0); 
}


inline std::string SOPTIX_Options::desc() const
{
    std::stringstream ss ;
    ss << "[SOPTIX_Options::desc" << std::endl ;
    ss << " _LEVEL " << _LEVEL << std::endl ; 
    ss << " Level " << Level() << std::endl ; 
    ss << " _optLevel " << _optLevel << std::endl ;
    ss << " _debugLevel " << _debugLevel << std::endl ;
    ss << " _exceptionFlags " << _exceptionFlags << std::endl ;
    ss << " _link_debugLevel " << _link_debugLevel << std::endl ;
    ss << Desc_moduleCompileOptions(moduleCompileOptions) ;  
    ss << Desc_pipelineCompileOptions(pipelineCompileOptions) ;  
    ss << Desc_pipelineLinkOptions(pipelineLinkOptions) ;  
    ss << "]SOPTIX_Options::desc" << std::endl ;
    std::string str = ss.str() ; 
    return str ; 
}

inline SOPTIX_Options::SOPTIX_Options()
    :
    _optLevel(   ssys::getenvvar(SOPTIX_Options_optLevel, "DEFAULT" ) ), 
    _debugLevel( ssys::getenvvar(SOPTIX_Options_debugLevel, "DEFAULT" ) ),
    _exceptionFlags( ssys::getenvvar(SOPTIX_Options_exceptionFlags, "STACK_OVERFLOW" ) ),
    _link_debugLevel( ssys::getenvvar(SOPTIX_Options_link_debugLevel, "DEFAULT" )),
    _numPayloadValues(2),  
    _numAttributeValues(2),  
    _maxTraceDepth(ssys::getenvunsigned(SOPTIX_Options_maxTraceDepth, 2))
{
    init();
}

inline void SOPTIX_Options::init()
{
    init_moduleCompileOptions();
    init_pipelineCompileOptions();
    init_pipelineLinkOptions();
}

/**
SOPTIX_Options::init_ModuleCompile
------------------------------------

moduleCompileOptions.numPayloadTypes
   Must be zero if OptixPipelineCompileOptions::numPayloadValues is not zero

**/

inline void SOPTIX_Options::init_moduleCompileOptions()
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

inline std::string SOPTIX_Options::Desc_moduleCompileOptions(const OptixModuleCompileOptions& module_compile_options )
{
    std::stringstream ss ; 
    ss  
       << "[SOPTIX_Options::Desc_moduleCompileOptions" << std::endl 
       << " module_compile_options.maxRegisterCount " << module_compile_options.maxRegisterCount  
       << " OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT " << OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT
       << std::endl 
       << " module_compile_options.optLevel         " << module_compile_options.optLevel  
       << " " << SOPTIX_OPT::OptimizationLevel_( module_compile_options.optLevel )
       << std::endl 
       << " module_compile_options.debugLevel       " << module_compile_options.debugLevel  
       << " " << SOPTIX_OPT::DebugLevel_( module_compile_options.debugLevel )
       << std::endl 
       << "]SOPTIX_Options::Desc_moduleCompileOptions" 
       << std::endl 
       ;   
    std::string str = ss.str() ; 
    return str ; 
}

inline void SOPTIX_Options::init_pipelineCompileOptions()
{
    int usesMotionBlur = 0 ; 

    OptixTraversableGraphFlags traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY ; 
    //OptixTraversableGraphFlags traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ; 
    //OptixTraversableGraphFlags traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING ;


    unsigned numPayloadValues = _numPayloadValues ; // in optixTrace call 
    unsigned numAttributeValues = _numAttributeValues ; 
    unsigned exceptionFlags = SOPTIX_OPT::ExceptionFlags( _exceptionFlags ) ;
    const char* pipelineLaunchParamsVariableName = "params";
    //unsigned usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ; 
    unsigned usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE ; 

    pipelineCompileOptions.usesMotionBlur        = usesMotionBlur ;
    pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags ; 
    pipelineCompileOptions.numPayloadValues      = numPayloadValues ;   // in optixTrace call
    pipelineCompileOptions.numAttributeValues    = numAttributeValues ; 
    pipelineCompileOptions.exceptionFlags        = exceptionFlags ; 
    pipelineCompileOptions.pipelineLaunchParamsVariableName = pipelineLaunchParamsVariableName ; 
    pipelineCompileOptions.usesPrimitiveTypeFlags = usesPrimitiveTypeFlags ;
}

inline std::string SOPTIX_Options::Desc_pipelineCompileOptions(const OptixPipelineCompileOptions& pipeline_compile_options )
{
    std::stringstream ss ;
    ss
       << "[SOPTIX_Options::Desc_pipelineCompileOptions" << std::endl
       << " pipeline_compile_options.numPayloadValues   " << pipeline_compile_options.numPayloadValues  << std::endl
       << " pipeline_compile_options.numAttributeValues " << pipeline_compile_options.numAttributeValues << std::endl
       << " pipeline_compile_options.exceptionFlags     " << pipeline_compile_options.exceptionFlags
       << SOPTIX_OPT::Desc_ExceptionFlags( pipeline_compile_options.exceptionFlags )
       << std::endl
       << "]SOPTIX_Options::Desc_pipelineCompileOptions" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}



inline void SOPTIX_Options::init_pipelineLinkOptions()
{
    OptixPayloadType* payloadType = nullptr ; 
    programGroupOptions.payloadType = payloadType ; 

#if OPTIX_VERSION <= 70600
    OptixCompileDebugLevel debugLevel = SOPTIX_OPT::DebugLevel(_link_debugLevel)  ;
    pipelineLinkOptions.debugLevel = debugLevel ;
#endif
    pipelineLinkOptions.maxTraceDepth = _maxTraceDepth ; 
}

inline std::string SOPTIX_Options::Desc_pipelineLinkOptions(const OptixPipelineLinkOptions& pipeline_link_options )
{
    std::stringstream ss ;
    ss
       << "[SOPTIX_Options::Desc_pipelineLinkOptions" << std::endl
       << " pipeline_link_options.maxTraceDepth   " << pipeline_link_options.maxTraceDepth  << std::endl
       << std::endl
#if OPTIX_VERSION <= 70600
       << " pipeline_link_options.debugLevel      " << pipeline_link_options.debugLevel  
       << " " << SOPTIX_OPT::DebugLevel_( pipeline_link_options.debugLevel )
#endif
       << "]SOPTIX_Options::Desc_pipelineLinkOptions" << std::endl
       ;
    std::string str = ss.str() ;
    return str ;
}


