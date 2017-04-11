#pragma once

#include "quad.h"

struct Part 
{
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    __device__ unsigned gtransformIdx(){ return q3.u.x ; }  //  gtransformIdx is 1-based, 0 meaning None 
    __device__ unsigned partType(){      return q2.u.w ; }  //  OptickCSG_t enum 
};




