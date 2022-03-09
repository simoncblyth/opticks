#pragma once


/**
PIP
=====

Aiming to keep this geometry independent 

This is used by CSGOptiX.cc and SBT.cc

**/

struct PIP
{
    unsigned max_trace_depth ; 
    unsigned num_payload_values ; 
    unsigned num_attribute_values ; 

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions program_group_options = {};

    OptixModule module = nullptr;

    OptixProgramGroup raygen_pg   = nullptr;
    OptixProgramGroup miss_pg     = nullptr;
    OptixProgramGroup hitgroup_pg = nullptr;

    OptixPipeline pipeline = nullptr;

    static OptixCompileDebugLevel        DebugLevel(const char* option); 
    static OptixCompileOptimizationLevel OptimizationLevel(const char* option) ; 

    static OptixPipelineCompileOptions CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues );
    static OptixProgramGroupOptions CreateProgramGroupOptions();

    static const char* CreateModule_debugLevel ; 
    static const char* CreateModule_optLevel ; 
    static OptixModule CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options );

    PIP(const char* ptx_path_); 
    const char* desc() const ; 

    void init(); 
    void createRaygenPG(const char* rg);
    void createMissPG(const char* ms);
    void createHitgroupPG(const char* is, const char* ch, const char* ah );

    static const char* linkPipeline_debugLevel ; 
    void linkPipeline(unsigned max_trace_depth);
    void configureStack(); 

}; 


