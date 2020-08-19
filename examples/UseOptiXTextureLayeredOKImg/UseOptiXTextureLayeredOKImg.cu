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
#include "UseOptiXTextureLayeredOKImg.h"

using namespace optix;

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

rtDeclareVariable(int4,   tex_param, , );
rtDeclareVariable(float4, tex_domain, , );

rtBuffer<uchar4, 3> out_buffer ; 

#ifdef TEX_BUFFER_CHECK
rtBuffer<uchar4, 3> tex_buffer ; 
RT_PROGRAM void readWrite()
{
    uchar4 val = tex_buffer[launch_index] ; 
    rtPrintf("//UseOptiXTextureLayeredOKImg.cu:readWrite.TEX_BUFFER_CHECK launch_index.xyz ( %u %u %u )   val ( %d %d %d %d ) \n", 
         launch_index.x, 
         launch_index.y, 
         launch_index.z, 
         val.x, 
         val.y, 
         val.z, 
         val.w 
       );

    out_buffer[launch_index] = val ; 
}
#else
RT_PROGRAM void readWrite()
{
    int texture_id = tex_param.w ; 
    float tx = float(launch_index.x);  
    float ty = float(launch_index.y);  
    int layer = launch_index.z ; 
    uchar4 val = rtTex2DLayered<uchar4>( texture_id, tx, ty, layer );

    rtPrintf("//UseOptiXTextureLayeredOKImg.cu:readWrite.fromTex tex_param (%d %d %d %d) launch_index.xyz ( %u %u %u )   val ( %d %d %d %d ) \n", 
         tex_param.x,
         tex_param.y,
         tex_param.z,
         tex_param.w,
         launch_index.x, 
         launch_index.y, 
         launch_index.z, 
         val.x, 
         val.y, 
         val.z, 
         val.w 
       );
    out_buffer[launch_index] = val ; 
}
#endif


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


