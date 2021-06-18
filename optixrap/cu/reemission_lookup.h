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

rtTextureSampler<float, 2>  reemission_texture ;
rtDeclareVariable(float4, reemission_domain, , );

static __device__ __inline__ float reemission_lookup(float u)
{
    float ui = u/reemission_domain.z + 0.5f ;   
    return tex2D(reemission_texture, ui, 0.5f );  // line 0
}

static __device__ __inline__ void reemission_check()
{
#ifdef WITH_PRINT
    float nm_a = reemission_lookup(0.0f); 
    float nm_b = reemission_lookup(0.5f); 
    float nm_c = reemission_lookup(1.0f); 
    rtPrintf("reemission_check nm_a %10.3f %10.3f %10.3f  \n",  nm_a, nm_b, nm_c );
#endif
}


