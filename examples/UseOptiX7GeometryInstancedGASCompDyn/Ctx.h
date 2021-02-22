#pragma once
#include <optix.h>

struct Ctx 
{
    Ctx(); 

    static OptixDeviceContext context ;
    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

};

