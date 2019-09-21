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

#include <optix_world.h>
#include <curand_kernel.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<curandState, 1>       rng_states ;

#define WITH_PRINT 1 


RT_PROGRAM void ORngTest()
{
    unsigned long long photon_id = launch_index.x ;  
    //unsigned int photon_offset = photon_id*4 ; 
 
    curandState rng = rng_states[photon_id];

    //photon_buffer[photon_offset+0] = make_float4( curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );

    float u = curand_uniform(&rng) ; 


#ifdef WITH_PRINT
    rtPrintf("ORngTest.cu:ORngTest  d %d  (%10.4f,%10.4f,%10.4f,%10.4f)  \n", photon_id, u,u,u,u  );
#endif
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



