#pragma once

#include <optix.h>
struct Properties ; 

struct Ctx 
{
    Properties* props  ; 
    static OptixDeviceContext context ;
    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 
    Ctx(); 

};

