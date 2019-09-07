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

rtDeclareVariable(unsigned int,  PNUMQUAD, , );
rtDeclareVariable(unsigned int,  RNUMQUAD, , );
rtDeclareVariable(unsigned int,  GNUMQUAD, , );

#include "cu/quad.h"

rtBuffer<float4>               genstep_buffer ;
rtBuffer<float4>               photon_buffer;
rtBuffer<short4>               record_buffer;     // 2 short4 take same space as 1 float4 quad
rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(unsigned int,  record_max, , );


RT_PROGRAM void eventTest()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ;   // 4

    rtPrintf("//eventTest %d \n", PNUMQUAD ); 


    unsigned int MAXREC = record_max ; 
    int slot_min = photon_id*MAXREC ; 

    int record_offset = 0 ; 
    for(int slot=0 ; slot < MAXREC ; slot++)
    {
         record_offset = (slot_min + slot)*RNUMQUAD ;
         record_buffer[record_offset+0] = make_short4(slot,slot,slot,slot) ;    // 4*int16 = 64 bits
         record_buffer[record_offset+1] = make_short4(slot,slot,slot,slot) ;    
    }  

    photon_buffer[photon_offset+0] = make_float4( 0.f , 0.f, 0.f, 0.f );
    photon_buffer[photon_offset+1] = make_float4( 1.f , 1.f, 1.f, 1.f );
    photon_buffer[photon_offset+2] = make_float4( 2.f , 2.f, 2.f, 2.f );
    photon_buffer[photon_offset+3] = make_float4( 3.f , 3.f, 3.f, 3.f );

    unsigned long long seqhis = 0ull ; 
    unsigned long long seqmat = 1ull ; 

    sequence_buffer[photon_id*2 + 0] = seqhis ; 
    sequence_buffer[photon_id*2 + 1] = seqmat ;  

}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}



