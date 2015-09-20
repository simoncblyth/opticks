#pragma once

#include "quad.h"
#include "rotateUz.h"

struct TorchStep
{
    int Id    ;
    int ParentId ;
    int MaterialIndex  ;
    int NumPhotons ;

    float3 x0 ;
    float  t0 ;

    float3 p0 ;
    float  sp ;
 
    float4 spare3 ; 
    float4 spare4 ; 
    float4 spare5 ; 

};


__device__ void tsload( TorchStep& ts, optix::buffer<float4>& genstep, unsigned int offset, unsigned int genstep_id)
{
    union quad ipmn, ccwv, mmmm  ;
 
    ipmn.f = genstep[offset+0];     
    ts.Id = genstep_id ; 
    ts.ParentId = ipmn.i.y ; 
    ts.MaterialIndex = ipmn.i.z ; 
    ts.NumPhotons = ipmn.i.w ; 

    float4 xt0 = genstep[offset+1];
    ts.x0 = make_float3(xt0.x, xt0.y, xt0.z );
    ts.t0 = xt0.w ; 

    float4 p0s = genstep[offset+2];
    ts.p0 = make_float3(p0s.x, p0s.y, p0s.z );
    ts.sp = p0s.w ; 
    
}


__device__ void tsdump( TorchStep& ts )
{
    rtPrintf("ts.Id %d ParentId %d MaterialIndex %d NumPhotons %d \n", 
       ts.Id, 
       ts.ParentId, 
       ts.MaterialIndex, 
       ts.NumPhotons 
       );

    rtPrintf("x0 %f %f %f  t0 %f \n", 
       ts.x0.x, 
       ts.x0.y, 
       ts.x0.z, 
       ts.t0 
       );
}


__device__ void tsdebug( TorchStep& ts )
{
     tsdump(ts);
}



__device__ void
generate_torch_photon(Photon& p, TorchStep& ts, curandState &rng)
{
      p.wavelength = 500.f ; 

      float theta = 1.f*M_PIf*curand_uniform(&rng);
      float sinTheta, cosTheta;
      sincosf(theta,&sinTheta,&cosTheta);

      float phi = 2.f*M_PIf*curand_uniform(&rng);
      float sinPhi, cosPhi;
      sincosf(phi,&sinPhi,&cosPhi);
	
      // calculate x,y, and z components of photon energy
      // (in coord system with primary particle direction 
      //  aligned with the z axis)
      // then rotate momentum direction back to global reference system  

      float3 photonMomentum = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 
      rotateUz(photonMomentum, ts.p0 );
      p.direction = photonMomentum ;

      // Determine polarization of new photon 
      // and rotate back to original coord system 

      float3 photonPolarization = make_float3( cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta);
      rotateUz(photonPolarization, ts.p0);
      p.polarization = photonPolarization ;

      p.time = ts.t0 ;
      p.position = ts.x0 ;
      p.weight = 1.0f ;

      p.flags.u.x = 0 ;
      p.flags.u.y = 0 ;
      p.flags.u.z = 0 ;
      p.flags.u.w = 0 ;

}


