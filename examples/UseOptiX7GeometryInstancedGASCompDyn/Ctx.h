#pragma once

#include <glm/glm.hpp>
#include <optix.h>

struct Params ; 
struct AS ; 

struct Ctx 
{
    Params*     params ; 
    CUdeviceptr d_param;
    static OptixDeviceContext context ;

    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

    Ctx(); 

    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_, unsigned cameratype );
    void setSize(unsigned width_, unsigned height_, unsigned depth_ );
    void setTop(const AS* top);

    void uploadParams();

};

