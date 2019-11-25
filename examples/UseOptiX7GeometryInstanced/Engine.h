#pragma once

#include <vector>

#include <optix.h>





#include "GAS.h"
#include "IAS.h"
#include "PIP.h"


// transitional class for adiabatic breakup of the monolith 
struct Engine
{
    int rc ; 

    GAS gas = {} ;
    IAS ias ;
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



