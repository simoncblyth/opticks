#pragma once

#include <vector>
#include <optix.h>
#include <glm/glm.hpp>
#include <vector_types.h>

struct Geo ; 
struct PIP ; 
struct SBT ; 

struct Engine
{
    int rc ; 

    PIP* pip ;   
    Geo* geo ; 
    SBT* sbt ; 

    static OptixDeviceContext context ;

    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

    unsigned width = 0u ; 
    unsigned height = 0u ; 
    unsigned depth = 0u ; 

    std::vector<uchar4> host_pixels ; 
    uchar4* device_pixels = nullptr ; 


    Engine(const char* ptx_path_, const char* spec); 
    int preinit(); 
    void init(); 

    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_); 
    void setSize(unsigned width_, unsigned height_, unsigned depth_); 
    void allocOutputBuffer(); 
    void launch(); 
    void download(); 
    void writePPM(const char* path); 
}; 



