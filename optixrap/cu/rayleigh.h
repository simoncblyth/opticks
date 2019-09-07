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

/**
http://www.nat.vu.nl/en/sec/atom/Publications/pdf/rayleigh.pdf

 Lord Rayleigh, Philos. Mag. 47, 375 (1899).

**/

#include "random.h"
#include "rotateUz.h"

// port:source/processes/optical/src/G4OpRayleigh.cc
// http://bugzilla-geant4.kek.jp/show_bug.cgi?id=207  Xin Qian patch


__device__ void rayleigh_scatter_align(Photon &p, curandState &rng)
{
#ifdef WITH_ALIGN_DEV_DEBUG
    rtPrintf("rayleigh_scatter_align p.direction (%.9g %.9g %.9g) \n", p.direction.x, p.direction.y, p.direction.z );
    rtPrintf("rayleigh_scatter_align p.polarization (%.9g %.9g %.9g) \n", p.polarization.x, p.polarization.y, p.polarization.z );
#endif

    float3 newDirection = make_float3(0.f,0.f,0.f) ; 
    float3 newPolarization = make_float3(0.f,0.f,0.f);

    bool looping(true) ;  // while block vars need global scope

    do 
    {
        float u0 = curand_uniform(&rng) ; 
        float u1 = curand_uniform(&rng) ; 
        float u2 = curand_uniform(&rng) ; 
        float u3 = curand_uniform(&rng) ;
        float u4 = curand_uniform(&rng) ;  

#ifdef WITH_ALIGN_DEV_DEBUG

        rtPrintf("rayleigh_scatter_align.do u_OpRayleigh:%.9g \n", u0);
        rtPrintf("rayleigh_scatter_align.do u_OpRayleigh:%.9g \n", u1);
        rtPrintf("rayleigh_scatter_align.do u_OpRayleigh:%.9g \n", u2);
        rtPrintf("rayleigh_scatter_align.do u_OpRayleigh:%.9g \n", u3);
        rtPrintf("rayleigh_scatter_align.do u_OpRayleigh:%.9g \n", u4);
#endif
        float cosTheta = u0 ;
        float sinTheta = sqrtf(1.0f-u0*u0);
       
        if(u1 < 0.5f ) cosTheta = -cosTheta ; 

        float sinPhi ; 
        float cosPhi ; 
        sincosf(2.f*M_PIf*u2,&sinPhi,&cosPhi);
	
        newDirection.x = sinTheta * cosPhi;
        newDirection.y = sinTheta * sinPhi;
        newDirection.z = cosTheta ;

        rotateUz(newDirection, p.direction );

        float constant = -dot(newDirection,p.polarization);

        newPolarization.x = p.polarization.x + constant*newDirection.x ;
        newPolarization.y = p.polarization.y + constant*newDirection.y ;
        newPolarization.z = p.polarization.z + constant*newDirection.z ;


        // There is a corner case, where the Newmomentum direction
        // is the same as oldpolariztion direction:
        // random generate the azimuthal angle w.r.t. Newmomentum direction
 
        if(length(newPolarization) == 0.f )
        {
            sincosf(2.f*M_PIf*u3,&sinPhi,&cosPhi);
	
            newPolarization.x = cosPhi ; 
            newPolarization.y = sinPhi ; 
            newPolarization.z = 0.f ; 

            rotateUz(newPolarization, newDirection);
        }
        else
        {
            // There are two directions which are perpendicular
            // to the new momentum direction
            if(u3 < 0.5f) newPolarization = -newPolarization ;
        }

        newPolarization = normalize(newPolarization);

        // simulate according to the distribution cos^2(theta)
        // where theta is the angle between old and new polarizations
        float doCosTheta = dot(newPolarization,p.polarization) ;  
        float doCosTheta2 = doCosTheta*doCosTheta ;

        looping = doCosTheta2 < u4 ; 

#ifdef WITH_ALIGN_DEV_DEBUG
        rtPrintf("rayleigh_scatter_align.do constant        (%.9g) \n", constant );
        rtPrintf("rayleigh_scatter_align.do newDirection    (%.9g %.9g %.9g) \n", newDirection.x, newDirection.y, newDirection.z );
        rtPrintf("rayleigh_scatter_align.do newPolarization (%.9g %.9g %.9g) \n", newPolarization.x, newPolarization.y, newPolarization.z );
        rtPrintf("rayleigh_scatter_align.do doCosTheta %.9g doCosTheta2 %.9g   looping %d   \n", doCosTheta, doCosTheta2, looping );
#endif

    } while ( looping ) ;

    p.direction = newDirection ;
    p.polarization = newPolarization ;
}


__device__ void rayleigh_scatter(Photon &p, curandState &rng)
{

#ifdef WITH_ALIGN_DEV_DEBUG
    rtPrintf("rayleigh_scatter\n");
#endif

    float3 newDirection ; 
    float3 newPolarization ; 
    float cosTheta ;

    do {
        // Try to simulate the scattered photon momentum direction
        // w.r.t. the initial photon momentum direction

        newDirection = uniform_sphere(&rng);
        rotateUz(newDirection, p.direction );

        // calculate the new polarization direction
        // The new polarization needs to be in the same plane as the new
        // momentum direction and the old polarization direction

        float constant = -dot(newDirection,p.polarization);
        newPolarization = p.polarization + constant*newDirection ;

        // There is a corner case, where the Newmomentum direction
        // is the same as oldpolariztion direction:
        // random generate the azimuthal angle w.r.t. Newmomentum direction
 
        if(length(newPolarization) == 0.f )
        {
            float sinPhi, cosPhi;
            sincosf(2.f*M_PIf*curand_uniform(&rng),&sinPhi,&cosPhi);
	
            newPolarization.x = cosPhi ; 
            newPolarization.y = sinPhi ; 
            newPolarization.z = 0.f ; 

            rotateUz(newPolarization, newDirection);
        }
        else
        {
            // There are two directions which are perpendicular
            // to the new momentum direction
            if(curand_uniform(&rng) < 0.5f) newPolarization = -newPolarization ;
        }

        newPolarization = normalize(newPolarization);
   
        // simulate according to the distribution cos^2(theta)
        // where theta is the angle between old and new polarizations
        cosTheta = dot(newPolarization,p.polarization) ; 
 
    } while ( cosTheta*cosTheta < curand_uniform(&rng)) ;

    p.direction = newDirection ;
    p.polarization = newPolarization ;
}



