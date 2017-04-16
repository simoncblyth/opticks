#pragma once

#include "quad.h"

struct Part 
{
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 




    __device__ unsigned transformIdx(){  return q3.u.w ; }  //   transformIdx is 1-based, 0 meaning None 
    __device__ unsigned gtransformIdx(){ return q3.u.x ; }  //  gtransformIdx is 1-based, 0 meaning None 


    //__device__ unsigned index(){         return q1.u.y ; }  //
    __device__ unsigned index(){         return q1.u.w ; }  //
    __device__ unsigned nodeIndex(){     return q3.u.w ; }  //   <-- clash with transformIdx
    __device__ unsigned boundary(){      return q1.u.z ; }  //   see ggeo-/GPmt
    __device__ unsigned typecode(){      return q2.u.w ; }  //  OptickCSG_t enum 



};




