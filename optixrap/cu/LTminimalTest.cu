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

#include "quad.h"
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void minimal()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*6 ; 

    union quad ipmn;
    ipmn.f = output_buffer[photon_offset+0];

    rtPrintf("RT %d\n", ipmn.i.w);

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();

    //unsigned long long photon_id = launch_index.x ;  
    //unsigned int photon_offset = photon_id*6 ; 
}



