/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include "quad.h"

struct Part 
{

    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    __device__ unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
    __device__ bool        complement() const { return q3.u.w & 0x80000000 ; }


    __device__ unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
    __device__ unsigned planeNum()      const { return q0.u.y ; } 

    __device__ void setPlaneIdx(unsigned idx){  q0.u.x = idx ; }
    __device__ void setPlaneNum(unsigned num){  q0.u.y = num ; }


    //__device__ unsigned index()   const {      return q1.u.y ; }  //
    __device__ unsigned index()     const {      return q1.u.w ; }  //
    //__device__ unsigned nodeIndex() const {      return q3.u.w ; }  //   <-- clash with transformIdx
    __device__ unsigned boundary()  const {      return q1.u.z ; }  //   see ggeo-/GPmt
    __device__ unsigned typecode()  const {      return q2.u.w ; }  //  OptickCSG_t enum 



};




