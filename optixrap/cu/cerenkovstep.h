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
    float preVelocity ; 

    /// the above first 4 quads are common to both CerenkovStep and ScintillationStep 

    float BetaInverse ; 
    float Pmin ; 
    float Pmax ; 
    float maxCos ; 
 
    float maxSin2 ;
    float MeanNumberOfPhotons1 ; 
    float MeanNumberOfPhotons2 ; 
    float postVelocity ; 
    //int   BialkaliMaterialIndex  ;

    // above are loaded parameters, below are derived from them

    float MeanNumberOfPhotonsMax ; 
    float3 p0 ;

};


__device__ void csload( CerenkovStep& cs, optix::buffer<float4>& cerenkov, unsigned int offset, unsigned int genstep_id)
{
    union quad ipmn, ccwv, mmmm  ;
 
    ipmn.f = cerenkov[offset+0];     
    //cs.Id = ipmn.i.x ; 
    cs.Id = genstep_id ; 
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
    cs.preVelocity = ccwv.f.w ;

    float4 bppm = cerenkov[offset+4] ;
    cs.BetaInverse = bppm.x ; 
    cs.Pmin = bppm.y ; 
    cs.Pmax = bppm.z ; 
    cs.maxCos= bppm.w ; 

    mmmm.f = cerenkov[offset+5] ;
    cs.maxSin2 = mmmm.f.x ; 
    cs.MeanNumberOfPhotons1 = mmmm.f.y ; 
    cs.MeanNumberOfPhotons2 = mmmm.f.z ; 
    cs.postVelocity = mmmm.f.w ; 
    //cs.BialkaliMaterialIndex = mmmm.i.w ; 


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

#ifdef WITH_PRINT
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

    rtPrintf("code %d  charge %f weight %f preVelocity %f postVelocity %f \n", 
       cs.code,
       cs.charge,
       cs.weight,
       cs.preVelocity,
       cs.postVelocity
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
#endif
}


__device__ void cscheck(CerenkovStep& cs)
{
    float nmlo = boundary_sample_reciprocal_domain(0.f);
    float nmmi = boundary_sample_reciprocal_domain(0.5f);   
    float nmhi = boundary_sample_reciprocal_domain(1.0f);

#ifdef WITH_PRINT
    rtPrintf("cscheck sample wavelength lo/mi/hi   %f %f %f \n", nmlo,nmmi,nmhi);   
#endif

    float4 prlo = boundary_lookup(nmlo, cs.MaterialIndex, 0);
    float4 prmi = boundary_lookup(nmmi, cs.MaterialIndex, 0);
    float4 prhi = boundary_lookup(nmhi, cs.MaterialIndex, 0);


#ifdef WITH_PRINT
    rtPrintf("cscheck sample rindex lo/mi/hi   %f %f %f \n", prlo.x,prmi.x,prhi.x);   
    rtPrintf("cscheck sample abslen lo/mi/hi   %f %f %f \n", prlo.y,prmi.y,prhi.y);   
    rtPrintf("cscheck sample scalen lo/mi/hi   %f %f %f \n", prlo.z,prmi.z,prhi.z);   
    rtPrintf("cscheck sample reempr lo/mi/hi   %f %f %f \n", prlo.w,prmi.w,prhi.w);   
#endif

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


//#define ALIGN_DEBUG 1 

__device__ void
generate_cerenkov_photon(Photon& p, CerenkovStep& cs, curandState &rng)
{
#ifdef ALIGN_DEBUG
    {
        float wavelength_0 = boundary_sample_reciprocal_domain(0.f); 
        float wavelength_1 = boundary_sample_reciprocal_domain(1.f); 
        rtPrintf(" wavelength_0 %10.5f wavelength_1 %10.5f \n", wavelength_0, wavelength_1 );  
    }
#endif


     float cosTheta ;
     float sin2Theta ;
     float wavelength ;
     float sampledRI ;
     float u ; 
     float u_maxSin2 ;

     // 
     //  sampling to get wavelength and cone angle 
     //
     // pick wavelength from uniform 1/wavelength distribution within the range, 
     // lookup refractive index
     // calculate cosTheta and sinTheta for the refractive index
     // 
     // issue: for some jpmt gensteps 
     //        cs.MaterialIndex 24 (Aluminium) with sampledRI 1.0 cs.BetaInverse 1.000001
     //        sin2Theta always dipping negative resulting in infinite loop
     //
     //        suspect bad gensteps, but meanwhile just constrain sin2Theta 
     //        with fmaxf( 0.0001f,
     //

     do {
    
        u = curand_uniform(&rng) ; 

        wavelength = boundary_sample_reciprocal_domain_v3(u);   

        float4 props = boundary_lookup(wavelength, cs.MaterialIndex, 0);

        sampledRI = props.x ; 

#ifdef ALIGN_DEBUG
        rtPrintf("gcp.u0 %10.5f wavelength %10.5f sampledRI %10.5f \n", u, wavelength, sampledRI  ); 
#endif

        cosTheta = cs.BetaInverse / sampledRI;  

        sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  // avoid going -ve 

        u = curand_uniform(&rng) ;

        u_maxSin2 = u*cs.maxSin2 ;

#ifdef ALIGN_DEBUG
        rtPrintf("gcp.u1 %10.5f u_maxSin2 %10.5f sin2Theta %10.5f \n", u, u_maxSin2, sin2Theta  ); 
#endif

  
      } while ( u_maxSin2 > sin2Theta);

      p.wavelength = wavelength ; 

      // Generate random position of photon on cone surface defined by Theta 

      
      u = curand_uniform(&rng) ; 

      float phi = 2.f*M_PIf*u ;
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

#ifdef ALIGN_DEBUG
      rtPrintf("gcp.u2   %10.5f phi %10.5f dir ( %10.5f %10.5f %10.5f ) pol ( %10.5f %10.5f %10.5f )  \n", u, phi, 
              p.direction.x, p.direction.y, p.direction.z ,
              p.polarization.x, p.polarization.y, p.polarization.z
           );   
#endif
     
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

#ifdef ALIGN_DEBUG
          rtPrintf("gcp.u3 %10.5f delta %10.5f NumberOfPhotons %10.5f  \n", fraction, delta, NumberOfPhotons  ) ;
#endif

          u = curand_uniform(&rng) ; 

          N = u * cs.MeanNumberOfPhotonsMax ;

#ifdef ALIGN_DEBUG
          rtPrintf("gcp.u4 %10.5f N %10.5f  \n", u, N  ); 
#endif


      }  while (N > NumberOfPhotons);



      float midVelocity = cs.preVelocity + fraction*( cs.postVelocity - cs.preVelocity )*0.5f ;  

      p.time = cs.t0 + delta / midVelocity ;

      p.position = cs.x0 + fraction * cs.DeltaPosition ; 

#ifdef ALIGN_DEBUG
      rtPrintf("gcp.post ( %10.5f %10.5f %10.5f : %10.5f )  \n", p.position.x, p.position.y, p.position.z, p.time  ); 
#endif


      p.weight = cs.weight ;

      p.flags.u.x = 0 ;
      p.flags.u.y = 0 ;
      p.flags.u.z = 0 ;
      p.flags.u.w = 0 ;

}


