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

#include "cu/quad.h"

//#include "wavelength_lookup.h"

//#define WITH_PRINT 1


rtTextureSampler<float, 2>  reemission_texture ;
rtDeclareVariable(float4, reemission_domain, , );

static __device__ __inline__ float reemission_lookup(float u)
{
    float ui = u/reemission_domain.z + 0.5f ;   
    return tex2D(reemission_texture, ui, 0.5f );  // line 0
}

static __device__ __inline__ void reemission_check()
{
    float nm_0 = reemission_lookup(0.000f); 
    float nm_1 = reemission_lookup(0.001f); 
    float nm_2 = reemission_lookup(0.002f); 
    float nm_3 = reemission_lookup(0.003f); 
    float nm_4 = reemission_lookup(0.004f); 

    float nm_5 = reemission_lookup(0.005f); 
    float nm_6 = reemission_lookup(0.006f); 
    float nm_7 = reemission_lookup(0.007f); 
    float nm_8 = reemission_lookup(0.008f); 
    float nm_9 = reemission_lookup(0.009f); 

#ifdef WITH_PRINT
    rtPrintf("reemission_check nm_0:9    %10.3f %10.3f %10.3f %10.3f %10.3f    %10.3f %10.3f %10.3f %10.3f %10.3f  \n", 
           nm_0, nm_1, nm_2, nm_3, nm_4,
           nm_5, nm_6, nm_7, nm_7, nm_9
         );
#endif


/*
OptiX 400 : forced different texture params, access needs debug... getting crazy:

     0. to 1. in 0.1 steps
     0. to 0.1 in 0.01 steps
     0. to 0.1 in 0.01 steps

reemission_check nm_0:9       415.124    180.000    180.000    180.000    180.000       180.000    180.000    180.000    180.000    180.000  
reemission_check nm_0:9       415.124    180.000    180.000    180.000    180.000       180.000    180.000    180.000    180.000    180.000  

*/

}

static __device__ __inline__ void reemission_lookup_test(float u0, float u1, float u_step)
{
    for(float u=u0 ; u < u1 ; u += u_step )
    {
        float nm0 = reemission_lookup(u);
        float nm1 = reemission_lookup(u);
#ifdef WITH_PRINT
        rtPrintf("  reemission_lookup(%10.3f) ->  %10.3f nm0   %10.3f nm1  \n", u, nm0, nm1 );
#endif
    }
}







RT_PROGRAM void textureTest()
{
   reemission_check();
   reemission_lookup_test(0.f, 1.f, 0.01f);
}


RT_PROGRAM void exception()
{
    //const unsigned int code = rtGetExceptionCode();
    rtPrintExceptionDetails();
}



