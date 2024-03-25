#pragma once

#include <optix.h>


struct SOPTIX_Params
{ 
    unsigned width  ;
    unsigned height ;
    uchar4*  pixels  ;

    float  tmin ; 
    float  tmax ; 

    unsigned cameratype ; 
    float3 eye;
    float3 U;  
    float3 V;  
    float3 W;  

    OptixTraversableHandle handle ;
};


