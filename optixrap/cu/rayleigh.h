#pragma once

#include "random.h"
#include "rotateUz.h"

// port:source/processes/optical/src/G4OpRayleigh.cc
// http://bugzilla-geant4.kek.jp/show_bug.cgi?id=207  Xin Qian patch


__device__ void rayleigh_scatter_align(Photon &p, curandState &rng)
{
#ifdef WITH_ALIGN_DEV_DEBUG
    rtPrintf("rayleigh_scatter_align p.direction (%g %g %g) \n", p.direction.x, p.direction.y, p.direction.z );
    rtPrintf("rayleigh_scatter_align p.polarization (%g %g %g) \n", p.polarization.x, p.polarization.y, p.polarization.z );
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
        rtPrintf("rayleigh_scatter_align.do u0:%g u1:%g u2:%g u3:%g u4:%g \n", u0,u1,u2,u3,u4);
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
        rtPrintf("rayleigh_scatter_align.do constant        (%g) \n", constant );
        rtPrintf("rayleigh_scatter_align.do newDirection    (%g %g %g) \n", newDirection.x, newDirection.y, newDirection.z );
        rtPrintf("rayleigh_scatter_align.do newPolarization (%g %g %g) \n", newPolarization.x, newPolarization.y, newPolarization.z );
        rtPrintf("rayleigh_scatter_align.do doCosTheta %g doCosTheta2 %g   looping %d   \n", doCosTheta, doCosTheta2, looping );
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

        float constant = -1.f/dot(newDirection,p.polarization);

        //newPolarization = newDirection + constant*p.polarization ;  // <-- bug revealed 2017/12/12 by alignment efforts
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



