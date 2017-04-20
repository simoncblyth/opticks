#pragma once

struct Prim 
{
    __device__ unsigned partOffset() const { return  prim.x ; } 
    __device__ unsigned numParts()   const { return  prim.y ; } 
    __device__ unsigned tranOffset() const { return  prim.z ; } 
    __device__ unsigned primFlag()   const { return  prim.w ; } 

    uint4 prim ; 

};


