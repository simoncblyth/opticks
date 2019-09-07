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

#include "Roots3And4.h"


using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
//rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtBuffer<float4>  output_buffer;

RT_PROGRAM void Roots3And4Test()
{
    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 

    float t_min = 4.f ; 

    double a[5] ; 
    a[4] = 1. ; 
    a[3] = -10. ;
    a[2] =  35. ;
    a[1] = -50. ;
    a[0] = 24. ; 

    double roots[4] ; 


    int num_roots = SolveQuartic(a, roots ); 

/*
    int num_roots = 4 ; 
    roots[0] = 1. ; 
    roots[1] = 2. ; 
    roots[2] = 3. ; 
    roots[3] = 4. ; 
*/


    float4 cand = make_float4(RT_DEFAULT_MAX) ;  
    int num_cand = 0 ;  
    if(num_roots > 0)
    {
        for(int i=0 ; i < num_roots ; i++)
        {
            if(roots[i] > t_min ) setByIndex(cand, num_cand++, roots[i]) ; 
        }   
    }
    float t_cand = num_cand > 0 ? fminf(cand) : t_min ;   // smallest root bigger than t_min

    rtPrintf("//Roots3And4Test photon_offset %d num_roots %d num_cand %d cand %10.3f %10.3f %10.3f %10.3f t_cand %10.3f  \n",
         photon_offset, num_roots, num_cand, cand.x, cand.y, cand.z, cand.w, t_cand );
    
    output_buffer[photon_offset+0] = make_float4(40.f, 40.f, 40.f, 40.f);
    output_buffer[photon_offset+1] = make_float4(41.f, 41.f, 41.f, 41.f);
    output_buffer[photon_offset+2] = make_float4(42.f, 42.f, 42.f, 42.f);
    output_buffer[photon_offset+3] = make_float4(43.f, 43.f, 43.f, 43.f);
}

RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();

    unsigned long long photon_id = launch_index.x ;  
    unsigned int photon_offset = photon_id*4 ; 
    
    output_buffer[photon_offset+0] = make_float4(-40.f, -40.f, -40.f, -40.f);
    output_buffer[photon_offset+1] = make_float4(-41.f, -41.f, -41.f, -41.f);
    output_buffer[photon_offset+2] = make_float4(-42.f, -42.f, -42.f, -42.f);
    output_buffer[photon_offset+3] = make_float4(-43.f, -43.f, -43.f, -43.f);

}


