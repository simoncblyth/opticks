#pragma once

#include "quad.h"
#include "rotateUz.h"
#include "TorchStepNPY.hpp"

struct TorchStep
{

    // (0) m_ctrl
    int Id    ;
    int ParentId ;
    int MaterialIndex  ;
    int NumPhotons ;

    // (1) m_post : position time 
    float3 x0 ;
    float  t0 ;

    // (2) m_dirw : direction weight 
    float3 p0 ;
    float  weight ;
 
    // (3) m_polw
    float3 pol ;
    float  wavelength ;

    // (4) m_zeaz : zenith, azimuth 
    float4 zeaz ; 

    // (5) m_beam : radius, ...  
    float4 beam ; 

    // transient: derived from beam.w
    unsigned int type ; 

    // transient: derived from beam.z
    unsigned int polz ; 


};


__device__ void tsload( TorchStep& ts, optix::buffer<float4>& genstep, unsigned int offset, unsigned int genstep_id)
{
    union quad ctrl, beam ;
 
    ctrl.f = genstep[offset+0];     
    ts.Id = genstep_id ; 
    ts.ParentId = ctrl.i.y ; 
    ts.MaterialIndex = ctrl.i.z ; 
    ts.NumPhotons = ctrl.i.w ; 

    float4 post = genstep[offset+1];
    ts.x0 = make_float3(post.x, post.y, post.z );
    ts.t0 = post.w ; 

    float4 dirw = genstep[offset+2];
    ts.p0 = make_float3(dirw.x, dirw.y, dirw.z );
    ts.weight = dirw.w ; 

    float4 polw = genstep[offset+3];
    ts.pol = make_float3(polw.x, polw.y, polw.z );
    ts.wavelength = polw.w ; 
 
    float4 zeaz = genstep[offset+4];
    ts.zeaz = make_float4(zeaz.x, zeaz.y, zeaz.z, zeaz.w );

    beam.f = genstep[offset+5];
    ts.beam = make_float4(beam.f.x, beam.f.y, 0.f, 0.f );
    ts.type = beam.u.w ; 
    ts.polz = beam.u.z ; 
    
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


/*

http://mathworld.wolfram.com/SpherePointPicking.html

To obtain points such that any small area on the sphere is expected to contain
the same number of points (right figure above), choose U and V to be random
variates on (0,1). Then

theta   =   2 pi U    
phi     =   acos( 2V-1 )


*/

__device__ void
generate_torch_photon(Photon& p, TorchStep& ts, curandState &rng)
{
      p.wavelength = ts.wavelength ; 
      p.time = ts.t0 ;
      p.weight = ts.weight ;
      p.flags.u.x = 0 ;
      p.flags.u.y = 0 ;
      p.flags.u.z = 0 ;
      p.flags.u.w = 0 ;

      float radius = ts.beam.x ; 

      float u1 = uniform(&rng, ts.zeaz.x, ts.zeaz.y);   // eg 0->0.5
      float u2 = uniform(&rng, ts.zeaz.z, ts.zeaz.w );  // eg 0->1

      float sinPhi, cosPhi;
      sincosf(2.f*M_PIf*u2,&sinPhi,&cosPhi);
	
      // calculate x,y, and z components of photon energy
      // (in coord system with primary particle direction 
      //  aligned with the z axis)
      // then rotate momentum direction back to global reference system  
      
      if( ts.type == T_DISC )
      { 
          // disc single direction emitter 

          p.direction = ts.p0 ;

          //float r = radius*u1 ; 
          float r = radius*sqrtf(u1) ;   // avoid bunching 

          float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f ); 
          rotateUz(discPosition, ts.p0);

          p.position = ts.x0 + discPosition ;

          float3 photonPolarization = make_float3( sinPhi, -cosPhi, 0.f); // adhoc
          rotateUz(photonPolarization, ts.p0);

          p.polarization = photonPolarization ;
      }
      else if( ts.type == T_SPHERE )
      {
          // fixed point uniform spherical emitter with configurable zenith, azimuth ranges
         
          float sinTheta, cosTheta;
          //sincosf(1.f*M_PIf*u1,&sinTheta,&cosTheta);  // distribution bunched at poles like this

          //  range zenithazimuth.x:.y
          cosTheta = 1.f - 2.0f*u1  ;     // 0:0.5 ->  1->0.
          sinTheta = sqrtf( 1.0f - cosTheta*cosTheta );

          float3 photonMomentum = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 
          rotateUz(photonMomentum, ts.p0 );

          float3 photonPolarization = make_float3( cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta); // perp to above
          rotateUz(photonPolarization, ts.p0);

          p.direction = photonMomentum ;
          p.polarization = photonPolarization ;
          p.position = ts.x0 ;

      }
      else if( ts.type == T_INVSPHERE )
      {
          // emit from surface of sphere directed to center 

          float sinTheta, cosTheta;
          cosTheta = 1.f - 2.0f*u1 ; 
          sinTheta = sqrtf( 1.0f - cosTheta*cosTheta );

          float3 spherePosition = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 
          p.direction = -spherePosition  ;

          float3 surfaceNormal = make_float3( 0.f, 0.f, 1.f );     // special casing known geometry, TODO: make an input    
          float3 photonPolarization = normalize(cross(p.direction, surfaceNormal));

          /*
          float3 photonPolarization = spherePosition.x > 0.f ? 
                                 make_float3( cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta)
                              :
                                 make_float3( sinTheta*sinPhi, -sinTheta*cosPhi, 0.f )   
                              ;
          rotateUz(photonPolarization, ts.p0);
          */ 

          p.polarization = photonPolarization ;
          p.position = ts.x0 + radius*spherePosition ;


      }
      else if( ts.type == T_REFLTEST )
      {
          // for reflection test need a uniform distribution of incident angle
          float sinTheta, cosTheta;
          sincosf(1.f*M_PIf*u1,&sinTheta,&cosTheta);  

          float3 spherePosition = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 
          p.direction = -spherePosition  ;

          float3 surfaceNormal = make_float3( 0.f, 0.f, 1.f );     // special casing known geometry, TODO: make an input    

          // S-polarized : ie perpendicular to plane of incidence
          // P-polarized : parallel to plane of incidence

          float3 photonPolarization = ts.polz == M_SPOL ? normalize(cross(p.direction, surfaceNormal)) : p.direction ;  

          p.polarization = photonPolarization ;
          p.position = ts.x0 + radius*spherePosition ;
      }



}


