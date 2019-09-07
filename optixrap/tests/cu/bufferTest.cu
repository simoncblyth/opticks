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

rtBuffer<float4>  genstep_buffer ;
rtBuffer<float4>  photon_buffer;
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


//#define WITH_PRINT 1


RT_PROGRAM void bufferTest()
{
    unsigned long long index = launch_index.x ;
    float4 val = genstep_buffer[index] ; 
    //rtPrintf("bufferTest.cu:bufferTest  llu %llu  (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
#ifdef WITH_PRINT
    rtPrintf("bufferTest.cu:bufferTest  d %d  (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
#endif
    photon_buffer[index] = val ; 
}


RT_PROGRAM void bufferTest_0()
{
#ifdef WITH_PRINT
    unsigned long long index = launch_index.x ;
    rtPrintf("bufferTest.cu:bufferTest_0  llu : %llu \n", index );
#endif
}

RT_PROGRAM void bufferTest_1()
{
    unsigned long long index = launch_index.x ;
    float4 val = genstep_buffer[index] ; 
#ifdef WITH_PRINT
    rtPrintf("bufferTest.cu:bufferTest_1  llu : %llu  gs (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
#endif
}

RT_PROGRAM void bufferTest_2()
{
    unsigned long long index = launch_index.x ;
    float4 val = photon_buffer[index] ; 
#ifdef WITH_PRINT
    rtPrintf("bufferTest.cu:bufferTest_2  llu : %llu  ph (%10.4f,%10.4f,%10.4f,%10.4f)  \n", index, val.x, val.y, val.z, val.w);
#endif
}

RT_PROGRAM void bufferTest_3()
{
    unsigned long long index = launch_index.x ;
    float4 val = genstep_buffer[index] ; 
#ifdef WITH_PRINT
    rtPrintf("bufferTest.cu:bufferTest_3  llu : %llu    \n", index );
#endif
    photon_buffer[index] = make_float4( val.x, val.y, val.z, val.w ) ; 
}





RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



