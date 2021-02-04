#pragma once

#include <vector>

#include <optix.h>
#include "GAS.h"
#include "PIP.h"


/**
Engine
==========

Breaking up a monolithic example, aiming to kick 
things into an integratable form.

* static context doesnt look like a good idea

* maybe hide the optix types in EngineImp to keep them out of this header
  for version switchability 

  * want the Engine to be above the detail of which OptiX version is being used

**/


struct Engine
{
    int rc ; 

    GAS gas = {} ;
    PIP pip ;   
    static OptixDeviceContext context ;

    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

    unsigned width = 0u ; 
    unsigned height = 0u ; 

    std::vector<uchar4> host_pixels ; 
    uchar4* device_pixels = nullptr ; 

    Engine(const char* ptx_path_); 
    int preinit(); 
    void init(); 

    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_); 
    void setSize(unsigned width_, unsigned height_); 
    void allocOutputBuffer(); 
    void launch(); 
    void download(); 
    void writePPM(const char* path); 
}; 



