#pragma once
/**
PIP : OptiX 7 Ray Trace Program Pipeline 
==========================================

Aiming to keep this geometry independent 

This is used by CSGOptiX.cc and SBT.cc

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



    PIP(const char* ptx_path_, const Properties* properties_ ); 
    const char* desc() const ; 

    void init(); 
    void createRaygenPG(const char* rg);
    void createMissPG(const char* ms);
    void createHitgroupPG(const char* is, const char* ch, const char* ah );


    static const char* linkPipeline_debugLevel ; 
    std::string Desc_PipelineLinkOptions(const OptixPipelineLinkOptions& pipeline_link_options ); 
    void linkPipeline(unsigned max_trace_depth);
    void configureStack(); 
}; 


