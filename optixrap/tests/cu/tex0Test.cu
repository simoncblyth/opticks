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

using namespace optix;


rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(int4,  tex_param, , );
rtBuffer<float,2>           out_buffer;

// contrast with texTest which uses float4 access to the texture



RT_PROGRAM void tex0Test()
{
    // texture indexing for (nx, ny)
    //      array type indexing:   0:nx-1 , 0:ny-1
    //      norm float indexing:   0:1-1/nx  , 0:1-1/ny

    int ix = int(launch_index.x) ; 
    int iy = int(launch_index.y) ; 

    float x = (float(ix)+0.5f)/float(launch_dim.x) ; 
    float y = (float(iy)+0.5f)/float(launch_dim.y) ; 
    
    int tex_id = tex_param.x ; 
    float val = rtTex2D<float>( tex_id, x, y ); 

    //rtPrintf("tex0Test (%d,%d) (%10.4f,%10.4f) -> %10.4f  \n", ix, iy, x, y, val);

    out_buffer[launch_index] = val ; 

}

RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}



