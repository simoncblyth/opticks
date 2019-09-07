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


__device__ float
uniform(curandState *s, const float &low, const float &high)
{
    return low + curand_uniform(s)*(high-low);
}

__device__ float3
uniform_sphere(curandState *s) 
{
    float theta = uniform(s, 0.0f, 2.f*M_PIf);
    float u = uniform(s, -1.0f, 1.0f);
    float c = sqrtf(1.0f-u*u);

    return make_float3(c*cosf(theta), c*sinf(theta), u); 
}





