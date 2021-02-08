#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "Binding.h"

struct Geo ; 

/**


Hmm maybe PIP_Builder too ? 

**/

struct PIP
{
    Geo* geo ; 
    unsigned num_gas ; 

    glm::vec3 eye = {} ; 
    glm::vec3 U = {} ; 
    glm::vec3 V = {} ; 
    glm::vec3 W = {} ; 
    float tmin = 0.f ; 
    float tmax = 1e16f ; 

    ///////////////////

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModule module = nullptr;

    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixPipeline pipeline = nullptr;

    CUdeviceptr  raygen_record;
    RayGenSbtRecord rg_sbt;

    CUdeviceptr miss_record;
    MissSbtRecord ms_sbt;

    CUdeviceptr hitgroup_record;
    HitGroupSbtRecord hg_sbt ;
 
    OptixShaderBindingTable sbt = {};

    ///////////////////

    static OptixPipelineCompileOptions CreateOptions(unsigned numPayloadValues, unsigned numAttributeValues );
    static OptixModule CreateModule(const char* ptx_path, OptixPipelineCompileOptions& pipeline_compile_options );

    PIP(const char* ptx_path_); 
    void upload(); 
    void init(); 
    void createProgramGroups(); 
    void linkPipeline(); 
    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_ ); 

    void createSbt();  
    void createRayGenSbt();  
    void createMissSbt();  
    void createHitGroupSbt();  

    void updateSbt();  
    void updateRayGenSbt();  
    void updateMissSbt();  
    void updateHitGroupSbt();  
}; 

