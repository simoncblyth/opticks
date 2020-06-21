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

#include "quad.h"
#include "random.h"

struct Genstep_G4Scintillation_1042
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
    float preVelocity ; 

    /// the above first 4 quads are common to both CerenkovStep and ScintillationStep 

    int   ScintillationType ;   
    float spare1 ;    
    float spare2 ;    
    float spare3 ;

    float ScintillationTime ;
    float ScintillationRiseTime ;
    float postVelocity ;
    float Other2 ;

    // above are loaded parameters, below are derived from them
    float3 p0 ;

    __device__ void load(optix::buffer<float4>& scintillation, unsigned int offset, unsigned int genstep_id) 
    {
        union quad ipmn, ccwv, ssss ;
     
        ipmn.f = scintillation[offset+0];     
        //int gentype = ipmn.i.x ;  

        Id = genstep_id ;    // usurp the encumbent gentype
        ParentId = ipmn.i.y ; 
        MaterialIndex = ipmn.i.z ; 
        NumPhotons = ipmn.i.w ; 

        float4 xt0 = scintillation[offset+1];
        x0 = make_float3(xt0.x, xt0.y, xt0.z );
        t0 = xt0.w ; 
        
        float4 dpsl = scintillation[offset+2] ;
        DeltaPosition = make_float3(dpsl.x, dpsl.y, dpsl.z );
        step_length = dpsl.w ; 

        ccwv.f = scintillation[offset+3] ;
        code = ccwv.i.x ;
        charge = ccwv.f.y ;
        weight = ccwv.f.z ;
        preVelocity = ccwv.f.w ;

        ssss.f = scintillation[offset+4] ;
        ScintillationType = ssss.i.x ; 
        spare1 = ssss.f.y ; 
        spare2 = ssss.f.z ; 
        spare3 = ssss.f.w ; 

        float4 ssoo = scintillation[offset+5] ;
        ScintillationTime = ssoo.x ; 
        ScintillationRiseTime = ssoo.y ; 
        postVelocity = ssoo.z ; 
        Other2 = ssoo.w ; 

        // derived qtys 
        p0 = normalize(DeltaPosition);
    }

    __device__ void dump()
    {
        rtPrintf("Genstep_G4Scintillation_1042 ss.Id %d ParentId %d MaterialIndex %d NumPhotons %d \n", 
           Id, 
           ParentId, 
           MaterialIndex, 
           NumPhotons 
           );

#ifdef WITH_PRINT

        rtPrintf("x0 %f %f %f  t0 %f \n", 
           x0.x, 
           x0.y, 
           x0.z, 
           t0 
           );

        rtPrintf("DeltaPosition %f %f %f  step_length %f  \n", 
           DeltaPosition.x, 
           DeltaPosition.y, 
           DeltaPosition.z,
           step_length
           ); 

        rtPrintf("code %d  charge %f weight %f preVelocity %f \n", 
           code,
           charge,
           weight,
           preVelocity
          );

        rtPrintf("ScintillationType %d spare1 %f spare2 %f spare3 %f \n", 
           ScintillationType,
           spare1,
           spare2,
           spare3
          );

        rtPrintf("ScintillationTime %f  ScintillationRiseTime %f postVelocity %f Other2 %f \n", 
           ScintillationTime,
           ScintillationRiseTime,
           postVelocity,
           Other2
          );

        rtPrintf("p0 %f %f %f  \n", 
           p0.x, 
           p0.y, 
           p0.z
           );

#endif
    }

    __device__ void check()
    {
#ifdef WITH_PRINT
        float nmlo = boundary_sample_domain(0.f);
        float nmmi = boundary_sample_domain(0.5f);   
        float nmhi = boundary_sample_domain(1.0f);

        rtPrintf("check sample wavelength lo/mi/hi   %f %f %f \n", nmlo,nmmi,nmhi);   
#endif
    }

    __device__ void debug()
    {
        dump();
        check();
    }

    __device__ void generate_photon(Photon& p, curandState& rng)
    {
        // TODO: need fast and slow versions of this texture, plus potentially handling multiple scintillator materials
        p.wavelength = reemission_lookup(curand_uniform(&rng));

        p.direction = uniform_sphere(&rng);

        p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));  // pol is random vector orthogonal to dir


        float fraction = (charge != 0.f )?  curand_uniform(&rng) : 1.f ;    

        float delta = fraction * step_length ;  

        float midVelocity = preVelocity + fraction*(postVelocity-preVelocity)*0.5f ;  

        float deltaTime = ScintillationRiseTime == 0.f ?
                               delta/midVelocity - ScintillationTime*logf(curand_uniform(&rng)) 
                               :
                               0.f     // TODO: port sample_time(ScintillationRiseTime, ScintillationTime)/single_exp/bi_exp
                               ;


        p.time = t0 + deltaTime ; 

        p.position = x0 + fraction * DeltaPosition ;  

        p.weight = weight ;

        p.flags.u.x = ScintillationType ;   // TODO: check probably will be overwritten, need to bitsqueeze it too   
        p.flags.u.y = 0 ;
        p.flags.u.z = 0 ;
        p.flags.u.w = 0 ;
    }


};





