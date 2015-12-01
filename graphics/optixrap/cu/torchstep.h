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




/*



*disc*
    Photons start from a point on the disc

*discaxial*
    Shoot target with disc beam from 26 directions: from every axis(6), quadrant(4*3=12) and octant(8) 
    Expanding radius beyond the size of the object is useful for finding geometry bugs 

*sphere*

    Torch photons start from the transformed "source" 
    and are emitted in the direction 
    of the transformed "target"  ie direction normalize(tgt - src )


*invsphere*
*refltest*
    Photons start from a position on the sphere and go in 
    direction towards the center of the sphere 
    where the "source" position provides the center of the sphere 
    (so in this case the target is not used)


*/




__device__ float3 get_direction_4(unsigned int idir, float delta)
{ 
     // rays from these directions failed to intersect with the 
     // symmetric prism of apex 90 degres until modified intersect_prism 
     // to avoid the infinities 
     float3 dir ; 
     switch(idir)
     {
        case 0:dir = make_float3( 1.f  ,  delta,  delta);break;  // +X
        case 1:dir = make_float3( delta,  1.f  ,  delta);break;  // +Y
        case 2:dir = make_float3( delta,  delta,  delta);break;  // +Z
        case 3:dir = make_float3( 1.   ,  1.f  ,  delta);break;  // +X+Y    
     }
     return normalize(dir) ; 
}


__device__ float3 get_direction_26(unsigned int idir)
{  
     float3 dir ; 
     switch(idir)
     {
        case 0:dir = make_float3( 1.f,  0.f,  0.f);break;  // +X
        case 1:dir = make_float3(-1.f,  0.f,  0.f);break;  // -X
        case 2:dir = make_float3( 0.f,  1.f,  0.f);break;  // +Y
        case 3:dir = make_float3( 0.f, -1.f,  0.f);break;  // -Y
        case 4:dir = make_float3( 0.f,  0.f,  1.f);break;  // +Z
        case 5:dir = make_float3( 0.f,  0.f, -1.f);break;  // -Z

        case 6:dir = make_float3(  1.f,  1.f,  0.f);break;  // +X+Y     XY quadrants
        case 7:dir = make_float3( -1.f,  1.f,  0.f);break;  // -X+Y
        case 8:dir = make_float3( -1.f, -1.f,  0.f);break;  // -X-Y
        case 9:dir = make_float3(  1.f, -1.f,  0.f);break;  // +X-Y

        case 10:dir = make_float3(  1.f,  0.f, 1.f);break;  // +X+Z     XZ quadrants
        case 11:dir = make_float3( -1.f,  0.f, 1.f);break;  // -X+Z
        case 12:dir = make_float3( -1.f,  0.f,-1.f);break;  // -X-Z
        case 13:dir = make_float3(  1.f,  0.f,-1.f);break;  // +X-Z

        case 14:dir = make_float3(  0.f,  1.f, 1.f);break;  // +Y+Z     YZ quadrants
        case 15:dir = make_float3(  0.f, -1.f, 1.f);break;  // -Y+Z
        case 16:dir = make_float3(  0.f, -1.f,-1.f);break;  // -Y-Z
        case 17:dir = make_float3(  0.f,  1.f,-1.f);break;  // +Y-Z

        case 18:dir = make_float3(  1.f,   1.f,  1.f);break;  // +X+Y+Z   8 corners of unit cube
        case 19:dir = make_float3( -1.f,   1.f,  1.f);break;  // -X+Y+Z     one flip
        case 20:dir = make_float3(  1.f,  -1.f,  1.f);break;  // +X-Y+Z      
        case 21:dir = make_float3(  1.f,   1.f, -1.f);break;  // +X+Y-Z     
        case 22:dir = make_float3( -1.f,  -1.f,  1.f);break;  // -X-Y+Z     two flip
        case 23:dir = make_float3( -1.f,   1.f, -1.f);break;  // -X+Y-Z 
        case 24:dir = make_float3(  1.f,  -1.f, -1.f);break;  // +X-Y-Z 
        case 25:dir = make_float3( -1.f,  -1.f, -1.f);break;  // -X-Y-Z     all flip  
     }
     return normalize(dir) ; 
}




__device__ void
generate_torch_photon(Photon& p, TorchStep& ts, curandState &rng)
{
      //p.wavelength = ts.wavelength ; 
      p.wavelength = source_lookup(curand_uniform(&rng));  // Planck black body source 6500K standard illuminant 

      p.time       = ts.t0 ;
      p.weight     = ts.weight ;

      p.flags.u.x = 0 ;
      p.flags.u.y = 0 ;
      p.flags.u.z = 0 ;
      p.flags.u.w = 0 ;


      float radius = ts.beam.x ; 
      float distance = ts.beam.y ; 

      float u1 = uniform(&rng, ts.zeaz.x, ts.zeaz.y);   // eg 0->0.5
      float u2 = uniform(&rng, ts.zeaz.z, ts.zeaz.w );  // eg 0->1

      float sinPhi, cosPhi;
      sincosf(2.f*M_PIf*u2,&sinPhi,&cosPhi);
	
      // calculate x,y, and z components of photon energy
      // (in coord system with primary particle direction 
      //  aligned with the z axis)
      // then rotate momentum direction back to global reference system  
      
      if( ts.type == T_DISC || ts.type == T_DISCLIN )
      { 
          // disc single direction emitter 
          //  http://mathworld.wolfram.com/DiskPointPicking.html

          p.direction = ts.p0 ;

          float r = ts.type == T_DISC ? radius*sqrtf(u1) : radius*u1    ;   
          // taking sqrt intended to avoid pole bunchung 

          float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f ); 
          rotateUz(discPosition, ts.p0);

          p.position = ts.x0 + discPosition ;

          float3 photonPolarization = make_float3( sinPhi, -cosPhi, 0.f); // adhoc
          rotateUz(photonPolarization, ts.p0);

          p.polarization = photonPolarization ;
      }
      else if( ts.type == T_DISCAXIAL )
      {
          unsigned long long photon_id = launch_index.x ;  
          float3 dir = get_direction_26( photon_id % 26 );
          //float3 dir = get_direction_4( photon_id % 4, 0.f );

          float r = radius*sqrtf(u1) ; // sqrt avoids pole bunchung 
          float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f ); 
          rotateUz(discPosition, dir);

          p.position = ts.x0 + distance*dir + discPosition ;
          p.direction = -dir ;
          p.polarization = make_float3(0.f, 0.f, 1.f );

      }
      else if( ts.type == T_SPHERE )
      {
          // fixed point uniform spherical emitter with configurable zenith, azimuth ranges
         
          float sinTheta, cosTheta;

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

          float3 surfaceNormal = make_float3( 0.f, 0.f, 1.f );     
          // TODO: generalize, see below T_REFLTEST

          float3 photonPolarization = normalize(cross(p.direction, surfaceNormal));

          p.polarization = photonPolarization ;
          p.position = ts.x0 + radius*spherePosition ;

      }
      else if( ts.type == T_REFLTEST )
      {
          // for reflection test need a uniform distribution of incident angle
          // (this leads to distribution bunched at poles)

          float sinTheta, cosTheta;
          sincosf(1.f*M_PIf*u1,&sinTheta,&cosTheta);  

          float3 spherePosition = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 
          p.direction = -spherePosition  ;

          float3 surfaceNormal = make_float3( 0.f, 0.f, 1.f );    
          // *surfaceNormal* used to setup pure polarizations for specific geometry of first intersections
          // TODO: generalize somehow

          // S-polarized : ie perpendicular to plane of incidence
          // P-polarized : parallel to plane of incidence

          float3 photonPolarization = ts.polz == M_SPOL ? normalize(cross(p.direction, surfaceNormal)) : p.direction ;  

          p.polarization = photonPolarization ;
          p.position = ts.x0 + radius*spherePosition ;
      }
      else if( ts.type == T_INVCYLINDER )
      {
           // for prism test it is convenient for emission in all directions constrained to a plane 

          float3 cylinderPosition = make_float3( radius*cosPhi, radius*sinPhi, distance*(1.f - 2.0f*u1) );  
          float3 cylinderDirection = make_float3( cosPhi, sinPhi, 0.f ) ;

          // rotateUz(cylinderPosition, ts.p0 );  
          // the above rotation is about wrong axis, so have to carefully select 
          // azimuthal phifrac to fit the geometry

          p.direction = -cylinderDirection ; 
          p.position = ts.x0 + cylinderPosition ;

          float3 surfaceNormal = ts.pol ; 
          float3 photonPolarization = ts.polz == M_SPOL ? normalize(cross(p.direction, surfaceNormal)) : p.direction ;  
          p.polarization = photonPolarization ;

          unsigned long long photon_id = launch_index.x ;  

          float tdelta = float(photon_id % 100)*0.03333f ; // ripple effect 300 mm/ns * 0.0333 ns = ~10 mm between ripples
          p.time = ts.t0 + tdelta ; 

          //p.wavelength = float(photon_id % 10)*70.f + 100.f ; // toothcomb wavelength distribution 

      }


}


