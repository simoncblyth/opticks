#pragma once

struct Prim 
{
    __device__ int partOffset() const { return  prim.x ; } 
    __device__ int numParts()   const { return  prim.y < 0 ? -prim.y : prim.y ; } 
    __device__ int tranOffset() const { return  prim.z ; } 
    __device__ int planOffset() const { return  prim.w ; } 
    __device__ int primFlag()   const { return  prim.y < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE  ; } 

    int4 prim ; 

};


