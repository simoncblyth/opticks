#pragma once

#include "random.h"
#include "rotateUz.h"

// port:source/processes/optical/src/G4OpRayleigh.cc
// http://bugzilla-geant4.kek.jp/show_bug.cgi?id=207  Xin Qian patch

__device__ void rayleigh_scatter(Photon &p, curandState &rng)
{
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

        float constant = -1.f/dot(newDirection,p.polarization);
        newPolarization = newDirection + constant*p.polarization ;

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



