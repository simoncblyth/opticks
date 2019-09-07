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

//#include "numquad.h"
#include "quad.h"

using namespace optix;

//rtDeclareVariable(float,         SPEED_OF_LIGHT, , );
rtDeclareVariable(unsigned int,  PNUMQUAD, , );
rtDeclareVariable(unsigned int,  RNUMQUAD, , );
rtDeclareVariable(unsigned int,  GNUMQUAD, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );


rtBuffer<float4>  genstep_buffer;
rtBuffer<float4>  photon_buffer;

RT_PROGRAM void seedTest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*PNUMQUAD ; 
    //rtPrintf("(seedTest) photon_id %d photon_offset %d \n", photon_id, photon_offset );

    // first 4 bytes of photon_buffer photon records is seeded with genstep_id 
    // this seeding is done by App::seedPhotonsFromGensteps
    union quad phead ;
    phead.f = photon_buffer[photon_offset+0] ;
    unsigned int genstep_id = phead.u.x ; 
    // getting crazy values for this in interop, photon_buffer being overwritten ?? 
    unsigned int genstep_offset = genstep_id*GNUMQUAD ; 

    //union quad ghead ; 
    //ghead.f = genstep_buffer[genstep_offset+0]; 

    rtPrintf("(seedTest) photon_id %d photon_offset %d genstep_id %d GNUMQUAD %d genstep_offset %d \n", photon_id, photon_offset, genstep_id, GNUMQUAD, genstep_offset  );

    //rtPrintf("ghead.i.x %d \n", ghead.i.x );
}


RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
    //photon_buffer[launch_index.x] = make_float4(-1.f, -1.f, -1.f, -1.f);
}

