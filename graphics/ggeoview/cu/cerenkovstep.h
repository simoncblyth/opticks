#pragma once
// porting from /usr/local/env/chroma_env/src/chroma/chroma/cuda/cerenkov.h

#include "quad.h"
#include "rotateUz.h"

struct CerenkovStep
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

    float BetaInverse ; 
    float Pmin ; 
    float Pmax ; 
    float maxCos ; 
 
    float maxSin2 ;
    float MeanNumberOfPhotons1 ; 
    float MeanNumberOfPhotons2 ; 
    int   BialkaliMaterialIndex  ;

    // above are loaded parameters, below are derived from them

    float MeanNumberOfPhotonsMax ; 
    float3 p0 ;

};


__device__ void csload( CerenkovStep& cs, optix::buffer<float4>& cerenkov, unsigned int offset )
{
    union quad ipmn, ccwv, mmmm  ;
 
    ipmn.f = cerenkov[offset+0];     
    cs.Id = ipmn.i.x ; 
    cs.ParentId = ipmn.i.y ; 
    cs.MaterialIndex = ipmn.i.z ; 
    cs.NumPhotons = ipmn.i.w ; 

    float4 xt0 = cerenkov[offset+1];
    cs.x0 = make_float3(xt0.x, xt0.y, xt0.z );
    cs.t0 = xt0.w ; 
    
    float4 dpsl = cerenkov[offset+2] ;
    cs.DeltaPosition = make_float3(dpsl.x, dpsl.y, dpsl.z );
    cs.step_length = dpsl.w ; 

    ccwv.f = cerenkov[offset+3] ;
    cs.code = ccwv.i.x ;
    cs.charge = ccwv.f.y ;
    cs.weight = ccwv.f.z ;
    cs.MeanVelocity = ccwv.f.w ;

    float4 bppm = cerenkov[offset+4] ;
    cs.BetaInverse = bppm.x ; 
    cs.Pmin = bppm.y ; 
    cs.Pmax = bppm.z ; 
    cs.maxCos= bppm.w ; 

    mmmm.f = cerenkov[offset+5] ;
    cs.maxSin2 = mmmm.f.x ; 
    cs.MeanNumberOfPhotons1 = mmmm.f.y ; 
    cs.MeanNumberOfPhotons2 = mmmm.f.z ; 
    cs.BialkaliMaterialIndex = mmmm.i.w ; 


    //  derived qtys

    cs.p0 = normalize(cs.DeltaPosition);
    cs.MeanNumberOfPhotonsMax = max(cs.MeanNumberOfPhotons1, cs.MeanNumberOfPhotons2);
}


__device__ void csdump( CerenkovStep& cs )
{
    rtPrintf("cs.Id %d ParentId %d MaterialIndex %d NumPhotons %d \n", 
       cs.Id, 
       cs.ParentId, 
       cs.MaterialIndex, 
       cs.NumPhotons 
       );

    rtPrintf("x0 %f %f %f  t0 %f \n", 
       cs.x0.x, 
       cs.x0.y, 
       cs.x0.z, 
       cs.t0 
       );

    rtPrintf("DeltaPosition %f %f %f  step_length %f  \n", 
       cs.DeltaPosition.x, 
       cs.DeltaPosition.y, 
       cs.DeltaPosition.z,
       cs.step_length
       ); 

    rtPrintf("code %d  charge %f weight %f MeanVelocity %f \n", 
       cs.code,
       cs.charge,
       cs.weight,
       cs.MeanVelocity
      );

    rtPrintf("BetaInverse %f  Pmin %f Pmax %f maxCos %f \n", 
       cs.BetaInverse,
       cs.Pmin,
       cs.Pmax,
       cs.maxCos
      );

    rtPrintf("maxSin2 %f  MeanNumberOfPhotons1 %f MeanNumberOfPhotons2 %f MeanNumberOfPhotonsMax %f \n", 
       cs.maxSin2,
       cs.MeanNumberOfPhotons1,
       cs.MeanNumberOfPhotons2,
       cs.MeanNumberOfPhotonsMax
      );

    rtPrintf("p0 %f %f %f  \n", 
       cs.p0.x, 
       cs.p0.y, 
       cs.p0.z
       );
}


__device__ void cscheck(CerenkovStep& cs)
{
    float nmlo = sample_reciprocal_domain(0.f);
    float nmmi = sample_reciprocal_domain(0.5f);   
    float nmhi = sample_reciprocal_domain(1.0f);

    rtPrintf("cscheck sample wavelength lo/mi/hi   %f %f %f \n", nmlo,nmmi,nmhi);   

    float4 prlo = wavelength_lookup(nmlo, cs.MaterialIndex);
    float4 prmi = wavelength_lookup(nmmi, cs.MaterialIndex);
    float4 prhi = wavelength_lookup(nmhi, cs.MaterialIndex);

    rtPrintf("cscheck sample rindex lo/mi/hi   %f %f %f \n", prlo.x,prmi.x,prhi.x);   
    rtPrintf("cscheck sample abslen lo/mi/hi   %f %f %f \n", prlo.y,prmi.y,prhi.y);   
    rtPrintf("cscheck sample scalen lo/mi/hi   %f %f %f \n", prlo.z,prmi.z,prhi.z);   
    rtPrintf("cscheck sample reempr lo/mi/hi   %f %f %f \n", prlo.w,prmi.w,prhi.w);   

/*
    float c[3];
    c[0] = cs.BetaInverse / r[0];  
    c[1] = cs.BetaInverse / r[1];  
    c[2] = cs.BetaInverse / r[2];  

    printf("cscheck sample cosTheta lo/mi/hi   %f %f %f \n",
           c[0],c[1],c[2]
         );   


    float s[3];
    s[0] = (1.0 - c[0])*(1.0 + c[0]);
    s[1] = (1.0 - c[1])*(1.0 + c[1]);
    s[2] = (1.0 - c[2])*(1.0 + c[2]);

    printf("cscheck sample sin^2Theta lo/mi/hi   %f %f %f \n",
           s[0],s[1],s[2]
         );   
 */

}

__device__ void csdebug( CerenkovStep& cs )
{
     csdump(cs);
     cscheck(cs);
}



__device__ void
generate_cerenkov_photon(Photon& p, CerenkovStep& cs, curandState &rng)
{
     float cosTheta ;
     float sin2Theta ;
     float wavelength ;
     float sampledRI ;

     // 
     //  sampling to get wavelength and cone angle 
     //
     // pick wavelength from uniform 1/wavelength distribution within the range, 
     // lookup refractive index
     // calculate cosTheta and sinTheta for the refractive index
     // 
     do {

        wavelength = sample_reciprocal_domain(curand_uniform(&rng));   

        float4 props = wavelength_lookup(wavelength, cs.MaterialIndex);
        //float4 props = make_float4( 1.33f , 1000.f , 2000.f, 0.f );  // ri/ab/sc/re  DEBUG

        sampledRI = props.x ; 

        cosTheta = cs.BetaInverse / sampledRI;  

        sin2Theta = (1.f - cosTheta)*(1.f + cosTheta);
    
      } while ( curand_uniform(&rng)*cs.maxSin2 > sin2Theta);


      p.wavelength = wavelength ; 


      // Generate random position of photon on cone surface defined by Theta 

      float phi = 2.f*M_PIf*curand_uniform(&rng);
      float sinPhi, cosPhi;
      sincosf(phi,&sinPhi,&cosPhi);
	
      // calculate x,y, and z components of photon energy
      // (in coord system with primary particle direction 
      //  aligned with the z axis)
      // then rotate momentum direction back to global reference system  

      float sinTheta = sqrt(sin2Theta);
      float3 photonMomentum = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 
      rotateUz(photonMomentum, cs.p0 );
      p.direction = photonMomentum ;

      // Determine polarization of new photon 
      // and rotate back to original coord system 

      float3 photonPolarization = make_float3( cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta);
      rotateUz(photonPolarization, cs.p0);
      p.polarization = photonPolarization ;

     
      float fraction ; 
      float delta ;
      float NumberOfPhotons ;  
      float N ;


      float DeltaN = (cs.MeanNumberOfPhotons1-cs.MeanNumberOfPhotons2) ; 
      do 
      {

          fraction = curand_uniform(&rng) ;

          delta = fraction  * cs.step_length ;

          NumberOfPhotons = cs.MeanNumberOfPhotons1 - fraction * DeltaN  ;

          N = curand_uniform(&rng) * cs.MeanNumberOfPhotonsMax ;

      }  while (N > NumberOfPhotons);



      p.time = cs.t0 + delta / cs.MeanVelocity ;

      p.position = cs.x0 + fraction * cs.DeltaPosition ; 

      p.weight = cs.weight ;

      p.flags.u.x = 0 ;
      p.flags.u.y = 0 ;
      p.flags.u.z = 0 ;
      p.flags.u.w = 0 ;



}


