#pragma once

#include "quad.h"

struct Part 
{
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 


    //__device__ unsigned transformIdx()  const { return q3.u.w ; }  //   transformIdx is 1-based, 0 meaning None 
    //__device__ unsigned gtransformIdx() const { return q3.u.x ; }  //  gtransformIdx is 1-based, 0 meaning None 
    __device__ unsigned gtransformIdx() const { return q3.u.w ; }  //  gtransformIdx is 1-based, 0 meaning None 

    __device__ unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
    __device__ unsigned planeNum()      const { return q0.u.y ; } 


    //__device__ unsigned index()   const {      return q1.u.y ; }  //
    __device__ unsigned index()     const {      return q1.u.w ; }  //
    //__device__ unsigned nodeIndex() const {      return q3.u.w ; }  //   <-- clash with transformIdx
    __device__ unsigned boundary()  const {      return q1.u.z ; }  //   see ggeo-/GPmt
    __device__ unsigned typecode()  const {      return q2.u.w ; }  //  OptickCSG_t enum 



};




