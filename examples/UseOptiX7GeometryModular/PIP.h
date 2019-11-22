#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Binding.h"


struct PIP
{
    const char* ptx_path = nullptr ; 
    size_t sizeof_log = 0 ; 
    char log[2048]; // For error reporting from OptiX creation functions


    glm::vec3 eye = {} ; 
    glm::vec3 U = {} ; 
    glm::vec3 V = {} ; 
    glm::vec3 W = {} ; 


    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    OptixProgramGroup raygen_prog_group   = nullptr;
    OptixProgramGroup miss_prog_group     = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixPipeline pipeline = nullptr;

    OptixShaderBindingTable sbt = {};


    PIP(const char* ptx_path_); 

    void upload(); 

    void init(); 
    void createModule(); 
    void createProgramGroups(); 
    void linkPipeline(); 

    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_); 
    void createShaderBindingTable();  
    void updateShaderBindingTable();  

}; 








