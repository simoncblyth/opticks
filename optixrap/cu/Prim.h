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

struct Prim 
{
    __device__ int partOffset() const { return  q0.i.x ; } 
    __device__ int numParts()   const { return  q0.i.y < 0 ? -q0.i.y : q0.i.y ; } 
    __device__ int tranOffset() const { return  q0.i.z ; } 
    __device__ int planOffset() const { return  q0.i.w ; } 
    __device__ int primFlag()   const { return  q0.i.y < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE ; } 

    quad q0 ; 

};


