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
using namespace optix;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

rtDeclareVariable(int4,   tex_param, , );
rtDeclareVariable(float4, tex_domain, , );

rtBuffer<float,3> out_buffer ; 


RT_PROGRAM void readWrite()
{
    int texture_id = tex_param.w ; 
    float tx = float(launch_index.x);  
    float ty = float(launch_index.y);  
    int layer = launch_index.z ; 
    float val = rtTex2DLayered<float>( texture_id, tx, ty, layer );

    rtPrintf("//UseOptiXTextureLayeredOK.cu:readWrite tex_param (%d %d %d %d) tex_domain (%f %f %f %f) launch_index.xyz ( %u %u %u )   val %10.3f \n", 
         tex_param.x,
         tex_param.y,
         tex_param.z,
         tex_param.w,
         tex_domain.x,
         tex_domain.y,
         tex_domain.z,
         tex_domain.w,
         launch_index.x, 
         launch_index.y, 
         launch_index.z, 
         val
       );

    out_buffer[launch_index] = val ; 
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


