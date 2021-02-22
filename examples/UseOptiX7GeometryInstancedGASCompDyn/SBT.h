#pragma once

#include <optix.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Binding.h"
struct Params ; 


/**
SBT
====

Geometry specifics live here

**/

struct Geo ; 

struct SBT 
{
    SBT( const PIP* pip_, Params* params_ ); 
    void setGeo(const Geo* geo); 

    const PIP*    pip ; 

    Params*    params ; 
    Raygen*    raygen ;
    Miss*      miss ;
    HitGroup*  hitgroup ;
 
    CUdeviceptr  d_raygen ;
    CUdeviceptr  d_miss ;
    CUdeviceptr  d_hitgroup ;

    OptixShaderBindingTable sbt = {};

    void init();  
    void createRaygen();  
    void createMiss();  
    void createHitgroup(const Geo* geo);

    void updateRaygen();  
    void updateMiss();  
    void updateHitgroup();  
};

