#pragma once

#include <optix.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "Binding.h"


/**
SBT
====

Geometry specifics live here

**/

struct Geo ; 


struct SBT 
{
    SBT( const PIP* pip_ ); 
    void setGeo(const Geo* geo); 


    float tmin = 0.f ; 
    float tmax = 1e16f ; 
    glm::vec3 eye = {} ; 
    glm::vec3 U = {} ; 
    glm::vec3 V = {} ; 
    glm::vec3 W = {} ; 
    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_ ); 

    const PIP* pip ; 

    CUdeviceptr       d_raygen ;
    CUdeviceptr       d_miss ;
    CUdeviceptr       d_hitgroup ;

    Raygen   raygen ;
    Miss     miss ;
    HitGroup hitgroup ;
 
    OptixShaderBindingTable sbt = {};

    void init();  
    void createRaygen();  
    void createMiss();  
    void createHitgroup(const Geo* geo);

    void updateRaygen();  
    void updateMiss();  
    void updateHitgroup();  

};

