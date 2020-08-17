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

#include "UseOptiXTexture.h"
#include <optix_world.h>
using namespace optix;

rtBuffer<float,3> tex_buffer ; 
rtBuffer<float,3> out_buffer ; 


rtTextureSampler<float, 3> tex_sampler ;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );



RT_PROGRAM void readWrite()
{
    unsigned x = launch_index.x ; 
    unsigned y = launch_index.y ; 
    unsigned z = launch_index.z ; 
    unsigned nx = launch_dim.x ; 
    unsigned ny = launch_dim.y ; 
    unsigned nz = launch_dim.z ; 

#ifdef FROM_BUF
    float val = tex_buffer[launch_index] ;  
#else
    float3 tex_coord = make_float3( float(x), float(y), float(z)); 
    float val = tex3D( tex_sampler, tex_coord.x, tex_coord.y, tex_coord.z );
#endif

    rtPrintf("//UseOptiXTexture.cu:readWrite launch_index.xyz ( %u %u %u ) launch_dim.xyz (%u %u %u )  val %10.3f \n", 
         x, 
         y, 
         z, 
         nx, 
         ny, 
         nz, 
         val
       );

    out_buffer[launch_index] = val ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


