#pragma once

#include <glm/glm.hpp>
#include <optix.h>

struct Params ; 

struct Ctx 
{
    Params*     params ; 
    CUdeviceptr d_param;

    Ctx(); 

    static OptixDeviceContext context ;
    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

    void uploadParams()
    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_)


};

