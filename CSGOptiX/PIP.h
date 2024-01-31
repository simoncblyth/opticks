#pragma once
/**
PIP : OptiX 7 Ray Trace Program Pipeline 
==========================================

Aiming to keep this geometry independent 

The pip(PIP) instance is instanciated in CSGOptiX::initPIP
and passed as ctor argument to SBT

**/

#include "plog/Severity.h"

struct Properties ; 

struct PIP
{
    static const plog::Severity LEVEL ; 
    static const int MAX_TRACE_DEPTH ; 

    unsigned max_trace_depth ; 
    unsigned num_payload_values ; 
    unsigned num_attribute_values ; 
    const Properties* properties ; 

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions program_group_options = {};

    OptixModule module = nullptr;

    OptixProgramGroup raygen_pg   = nullptr;
    OptixProgramGroup miss_pg     = nullptr;
    OptixProgramGroup hitgroup_pg = nullptr;

    OptixPipeline pipeline = nullptr;


    static bool OptiXVersionIsSupported(); 


    static const char*                 CreatePipelineOptions_exceptionFlags ; 
    static OptixPipelineCompileOptions CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues );
    static OptixProgramGroupOptions CreateProgramGroupOptions();

    static const char* CreateModule_debugLevel ; 
    static const char* CreateModule_optLevel ; 

    static std::string Desc(); 
    static std::string Desc_ModuleCompileOptions(const OptixModuleCompileOptions& module_compile_options ); 
    static std::string Desc_PipelineCompileOptions(const OptixPipelineCompileOptions& pipeline_compile_options ); 

    static OptixModule CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options );
    void destroyModule(); 




    PIP(const char* ptx_path_, const Properties* properties_ ); 
    ~PIP(); 

    const char* desc() const ; 

    void init(); 
    void destroy(); 

    static constexpr const char* RG_DUMMY = "__raygen__rg_dummy" ; 
    static constexpr const char* RG = "__raygen__rg" ; 
    static constexpr const char* MS = "__miss__ms" ; 
    static constexpr const char* IS = "__intersection__is" ; 
    static constexpr const char* CH = "__closesthit__ch" ; 
    static constexpr const char* AH = "__anyhit__ah" ; 

    void createRaygenPG();
    void destroyRaygenPG();

    void createMissPG();
    void destroyMissPG();

    void createHitgroupPG();
    void destroyHitgroupPG();


    static const char* linkPipeline_debugLevel ; 
    std::string Desc_PipelineLinkOptions(const OptixPipelineLinkOptions& pipeline_link_options ); 
    void linkPipeline(unsigned max_trace_depth);
    void destroyPipeline(); 


    void configureStack(); 
}; 


