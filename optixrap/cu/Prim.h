#pragma once

#include "quad.h"

struct Prim 
{
    __device__ int partOffset() const { return  q0.i.x ; } 
    __device__ int numParts()   const { return  q0.i.y < 0 ? -q0.i.y : q0.i.y ; } 
    __device__ int tranOffset() const { return  q0.i.z ; } 
    __device__ int planOffset() const { return  q0.i.w ; } 
    __device__ int primFlag()   const { return  q0.i.y < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE ; } 

    quad q0 ; 

};


