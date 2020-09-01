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
#include <optixu/optixu_math_namespace.h>
#include "UseOptiXTextureLayeredOKImg.h"

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtBuffer<uchar4, 2> out_buffer ; 

rtDeclareVariable(int4,   tex_param, , );
rtDeclareVariable(float4, tex_domain, , );


#ifdef TEX_BUFFER_CHECK
rtBuffer<uchar4, 3> tex_buffer ; 
RT_PROGRAM void readWrite()
{
    uchar4 val = tex_buffer[launch_index] ; 
#ifdef DUMP
    rtPrintf("//UseOptiXTextureLayeredOKImg.cu:readWrite.TEX_BUFFER_CHECK launch_index.xyz ( %u %u %u )   val ( %d %d %d %d ) \n", 
         launch_index.x, 
         launch_index.y, 
         launch_index.z, 
         val.x, 
         val.y, 
         val.z, 
         val.w 
       );
#endif

    out_buffer[launch_index] = val ; 
}
#else
RT_PROGRAM void readWrite()
{
    float2 d = make_float2(launch_index)/make_float2(launch_dim) ; 
    int texture_id = tex_param.w ; 
    uchar4 val = rtTex2D<uchar4>( texture_id, d.x, d.y );
    //uchar4 val = make_uchar4( 255, 0, 0, 255); 

#ifdef DUMP
    rtPrintf("//UseOptiXTextureLayeredOKImg.cu:readWrite.fromTex tex_param (%d %d %d %d) launch_index.xy ( %u %u )   val ( %d %d %d %d ) \n", 
         tex_param.x,
         tex_param.y,
         tex_param.z,
         tex_param.w,
         launch_index.x, 
         launch_index.y, 
         val.x, 
         val.y, 
         val.z, 
         val.w 
       );
#endif

    out_buffer[launch_index] = val ; 
}
#endif


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}


