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
    HitGroup*     check ;
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};


    SBT( const PIP* pip_ ); 
    void setGeo(const Geo* geo); 

    void init();  

    void createRaygen();  
    void updateRaygen();  

    void createMiss();  
    void updateMiss();  

    void createHitgroup(const Geo* geo);
    void checkHitgroup(); 


    template <typename T>
    static T* UploadArray(const T* array, unsigned num_items ) ; 

    template <typename T>
    static T* DownloadArray(const T* array, unsigned num_items ) ; 



};

