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

#include "UseOptiXTextureLayered.h"
#include <optix_world.h>
using namespace optix;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

#ifdef FROM_BUF
rtBuffer<float,3> tex_buffer ; 
#endif

//rtTextureSampler<float, 2> tex_sampler ;
rtDeclareVariable(int4,  tex_param, , );

rtBuffer<float,3> out_buffer ; 


RT_PROGRAM void readWrite()
{
    int texture_id = tex_param.x ; 
    float tx = float(launch_index.x);  
    float ty = float(launch_index.y);  
    int layer = launch_index.z ; 

#ifdef FROM_BUF
    float val = tex_buffer[launch_index] ;  
#else
    float val = rtTex2DLayered<float>( texture_id, tx, ty, layer );
#endif

    rtPrintf("//UseOptiXTextureLayered.cu:readWrite tex_param (%d %d %d %d) launch_index.xyz ( %u %u %u ) tx ty %7.3f %7.3f lay %d    val %10.3f \n", 
         tex_param.x,
         tex_param.y,
         tex_param.z,
         tex_param.w,
         launch_index.x, 
         launch_index.y, 
         launch_index.z, 
         tx,
         ty,
         layer,  
         val
       );

    out_buffer[launch_index] = val ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


