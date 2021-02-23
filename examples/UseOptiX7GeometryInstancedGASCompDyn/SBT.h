#pragma once

#include <optix.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "Binding.h"


/**
SBT : RG,MS,HG program data preparation 
===========================================

**/

struct Geo ; 

struct SBT 
{
    const PIP*    pip ; 
    Raygen*       raygen ;
    Miss*         miss ;
    HitGroup*     hitgroup ;
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};



    SBT( const PIP* pip_ ); 
    void setGeo(const Geo* geo); 

    void init();  
    void createRaygen();  
    void createMiss();  
    void createHitgroup(const Geo* geo);

    void updateRaygen();  
    void updateMiss();  
    void updateHitgroup();  

    template <typename T>
    static T* UploadArray(const T* array, unsigned num_items ) ; 

};

