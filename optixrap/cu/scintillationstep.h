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

#pragma once
// porting from  /usr/local/env/chroma_env/src/chroma/chroma/cuda/scintillation.h

#include "quad.h"
#include "random.h"

struct ScintillationStep
{
    int Id    ;
    int ParentId ;
    int MaterialIndex  ;
    int NumPhotons ;

    float3 x0 ;
    float  t0 ;

    float3 DeltaPosition ;
    float  step_length ;

    int code; 
    float charge ;
    float weight ;
    float MeanVelocity ; 

    /// the above first 4 quads are common to both CerenkovStep and ScintillationStep 

    int scnt ;
    float slowerRatio ;   
    float slowTimeConstant ;    
    float slowerTimeConstant ;

    float ScintillationTime ;
    float ScintillationIntegralMax ;
    float Other1 ;
    float Other2 ;

    // above are loaded parameters, below are derived from them

    float3 p0 ;
};

__device__ void ssload( ScintillationStep& ss, optix::buffer<float4>& scintillation, unsigned int offset, unsigned int genstep_id)
{
    union quad ipmn, ccwv, ssss ;
 
    ipmn.f = scintillation[offset+0];     
    //ss.Id = ipmn.i.x ; 
    ss.Id = genstep_id ; 
    ss.ParentId = ipmn.i.y ; 
    ss.MaterialIndex = ipmn.i.z ; 
    ss.NumPhotons = ipmn.i.w ; 

    float4 xt0 = scintillation[offset+1];
    ss.x0 = make_float3(xt0.x, xt0.y, xt0.z );
    ss.t0 = xt0.w ; 
    
    float4 dpsl = scintillation[offset+2] ;
    ss.DeltaPosition = make_float3(dpsl.x, dpsl.y, dpsl.z );
    ss.step_length = dpsl.w ; 

    ccwv.f = scintillation[offset+3] ;
    ss.code = ccwv.i.x ;
    ss.charge = ccwv.f.y ;
    ss.weight = ccwv.f.z ;
    ss.MeanVelocity = ccwv.f.w ;

    ssss.f = scintillation[offset+4] ;
    ss.scnt = ssss.i.x ; 
    ss.slowerRatio = ssss.f.y ; 
    ss.slowTimeConstant = ssss.f.z ; 
    ss.slowerTimeConstant = ssss.f.w ; 

    float4 ssoo = scintillation[offset+5] ;
    ss.ScintillationTime = ssoo.x ; 
    ss.ScintillationIntegralMax = ssoo.y ; 
    ss.Other1 = ssoo.z ; 
    ss.Other2 = ssoo.w ; 

    // derived qtys 

    ss.p0 = normalize(ss.DeltaPosition);

}


__device__ void ssdump( ScintillationStep& ss )
{
    rtPrintf("ss.Id %d ParentId %d MaterialIndex %d NumPhotons %d \n", 
       ss.Id, 
       ss.ParentId, 
       ss.MaterialIndex, 
       ss.NumPhotons 
       );

#ifdef WITH_PRINT

    rtPrintf("x0 %f %f %f  t0 %f \n", 
       ss.x0.x, 
       ss.x0.y, 
       ss.x0.z, 
       ss.t0 
       );

    rtPrintf("DeltaPosition %f %f %f  step_length %f  \n", 
       ss.DeltaPosition.x, 
       ss.DeltaPosition.y, 
       ss.DeltaPosition.z,
       ss.step_length
       ); 

    rtPrintf("code %d  charge %f weight %f MeanVelocity %f \n", 
       ss.code,
       ss.charge,
       ss.weight,
       ss.MeanVelocity
      );

    rtPrintf("scnt %d slowerRatio %f slowTimeConstant %f slowerTimeConstant %f \n", 
       ss.scnt,
       ss.slowerRatio,
       ss.slowTimeConstant,
       ss.slowerTimeConstant
      );

    rtPrintf("ScintillationTime %f  ScintillationIntegralMax %f Other1 %f Other2 %f \n", 
       ss.ScintillationTime,
       ss.ScintillationIntegralMax,
       ss.Other1,
       ss.Other2
      );

    rtPrintf("p0 %f %f %f  \n", 
       ss.p0.x, 
       ss.p0.y, 
       ss.p0.z
       );

#endif
}


__device__ void sscheck(ScintillationStep& ss)
{
#ifdef WITH_PRINT
    float nmlo = boundary_sample_domain(0.f);
    float nmmi = boundary_sample_domain(0.5f);   
    float nmhi = boundary_sample_domain(1.0f);

    rtPrintf("sscheck sample wavelength lo/mi/hi   %f %f %f \n", nmlo,nmmi,nmhi);   
#endif
}


__device__ void ssdebug( ScintillationStep& ss )
{
    ssdump(ss);
    sscheck(ss);
    //reemission_check();
}




__device__ void
generate_scintillation_photon(Photon& p, ScintillationStep& ss, curandState& rng)
{
    float ScintillationTime = ss.ScintillationTime ; 
    if(ss.scnt == 2)
    {
        ScintillationTime = ss.slowTimeConstant ;
        if(curand_uniform(&rng) < ss.slowerRatio)
        {   
            ScintillationTime = ss.slowerTimeConstant ; 
        }  
    }

    // no materialIndex input to reemission_lookup as both scintillators share same CDF 
    p.wavelength = reemission_lookup(curand_uniform(&rng));

    p.direction = uniform_sphere(&rng);

    p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));  // pol is random vector orthogonal to dir

    float fraction = (ss.charge != 0.f )?  curand_uniform(&rng) : 1.f ;     // int charge ?

    float delta = fraction * ss.step_length ;  

    float deltaTime = delta/ss.MeanVelocity - ScintillationTime*logf(curand_uniform(&rng)) ; // hmm negative infinity potential here 

    p.time = ss.t0 + deltaTime ; 

    p.position = ss.x0 + fraction * ss.DeltaPosition ; 

    p.weight = ss.weight ;

    p.flags.u.x = 0 ;
    p.flags.u.y = 0 ;
    p.flags.u.z = 0 ;
    p.flags.u.w = 0 ;


}



