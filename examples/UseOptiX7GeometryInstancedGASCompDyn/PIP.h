#pragma once


/**
PIP
=====

Aiming to keep this geometry independent 

**/

struct PIP
{
    unsigned max_trace_depth ; 

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixProgramGroupOptions program_group_options = {};

    OptixModule module = nullptr;

    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixPipeline pipeline = nullptr;

    static OptixPipelineCompileOptions CreatePipelineOptions(unsigned numPayloadValues, unsigned numAttributeValues );
    static OptixProgramGroupOptions CreateProgramGroupOptions();
    static OptixModule CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options );

    PIP(const char* ptx_path_); 

    void init(); 
    void createRaygenPG(const char* rg);
    void createMissPG(const char* ms);
    void createHitgroupPG(const char* is, const char* ch, const char* ah );

    void linkPipeline(unsigned max_trace_depth);
}; 


