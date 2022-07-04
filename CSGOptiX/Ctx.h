#pragma once
/**
Ctx : instanciation creates OptiX 7 optixDeviceContext and populates Properties  
==================================================================================

**/

#include <optix.h>
#include <string>
#include "plog/Severity.h"

struct Properties ; 

struct Ctx 
{
    static const plog::Severity LEVEL ; 
    static OptixDeviceContext context ;
    static void log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

    Properties* props  ; 
    Ctx(); 

    std::string desc() const ; 
};

