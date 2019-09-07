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

#include "cu/boundary_lookup.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtBuffer<float4,2>  out_buffer;


RT_PROGRAM void boundaryLookupTest_ijk()
{
    int w = int(launch_index.x) ;
    int i = int(launch_index.y) ;
    float nm = boundary_domain.x + w*boundary_domain.z ;

    uint2 out_index ; 
    out_index.x = w ;  

    for(unsigned j=0 ; j < BOUNDARY_NUM_MATSUR ; j++){
    for(unsigned k=0 ; k < BOUNDARY_NUM_FLOAT4 ; k++)
    {
        out_index.y = boundary_lookup_ijk(i, j, k );  
        out_buffer[out_index] = boundary_lookup(nm, i, j, k ) ; 
    }
    }
}

RT_PROGRAM void boundaryLookupTest()
{
    int w = int(launch_index.x) ;
    int i = int(launch_index.y) ;
    float nm = boundary_domain.x + w*boundary_domain.z ;

    uint2 out_index ; 
    out_index.x = w ;  

    unsigned nj = BOUNDARY_NUM_MATSUR ;
    unsigned nk = BOUNDARY_NUM_FLOAT4 ;

    for(unsigned j=0 ; j < nj ; j++)
    {
        unsigned line = i*nj + j ;  
        for(unsigned k=0 ; k < nk ; k++)
        {
           out_index.y = boundary_lookup_linek(line, k );  
           out_buffer[out_index] = boundary_lookup(nm, line, k ) ; 
        }
    }
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}

/*

       #float4
            |     ___ wavelength samples
            |    /
   (123, 4, 2, 39, 4)
    |    |          \___ float4 props        
  #bnd   | 
         |
    omat/osur/isur/imat  
*/


