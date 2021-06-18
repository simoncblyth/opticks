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


#include <curand_kernel.h>
#include <optix_world.h>

using namespace optix;

//  rng_states rng_skipahead
#include "ORng.hh"
#include "reemission_lookup.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float>  out_buffer;

RT_PROGRAM void reemissionTest()
{
    unsigned long long photon_id = launch_index.x ;
    curandState rng = rng_states[photon_id];
    float u = curand_uniform(&rng);  
    float wavelength = reemission_lookup(u);   
    out_buffer[photon_id] = wavelength ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



