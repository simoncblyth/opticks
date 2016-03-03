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
    unsigned int mode ; 


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
    ts.mode = beam.u.z ; 
    
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
*refltest T_REFLTEST*

    Photons start from a position on the sphere and go in 
    direction towards the center of the sphere 
    where the "source" position provides the center of the sphere 
    (so in this case the target is not used)

*discIntersectSphere T_DISC_INTERSECT_SPHERE*

    Photons start from position on a disc canonically centered at [0,0,+600] 
    and all travel in same -Z direction [0,0,-1]. 
    They are incident on a sphere of the same radius as the disc. 

    For definiteness consider spherical coordinate system with 
    photons incident from above the Z pole

         x_sphere = r sin(th) cos(ph)
         y_sphere = r sin(th) sin(ph)
         z_sphere = r cos(th)

    At the point of intersection the surface normal is given by 

         surface_normal = [x_sphere, y_sphere, z_sphere ]  
  
    Plane of incidence contains the photon direction (actually the plane wave k vector)
    and the surface normal. 

    Cross product of the photon direction and surface normal is perpendicular
    to the plane of incidence, this corresponds to the S-polarized direction.

            norm_( z ^ surface_normal ) 

         =  norm_( [ -y , x,  0 ]  )     

         = [ -sin(ph), cos(ph), 0 ]


    Where y and x are the coordinates generated on the original disc, using
    the aligned coordinate systems of sphere and disc

    A vector perpendicular the above and the photon direction  
    is within the plane of incidence and defines the P-polarized direction.

             norm_( [  x,  y,  0 ] )     

          = [  cos(ph), sin(ph), 0 ]

    The spherical geometry and alighment of the source disc and sphere
    means that it is straightforward to arrange S-pol and P-pol without 
    needing to calculate the intersect.


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
      p.wavelength = ts.wavelength > 50. ? ts.wavelength : source_lookup(curand_uniform(&rng));  // Planck black body source 6500K standard illuminant 

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

          // huh, bow count numbers for this adhoc polarization are "by accident" 
          // matching those for the below careful setup of S-polarization by 
          // working out the sphere surface normal 

          p.polarization = photonPolarization ;

      }
      else if( ts.type == T_RING )
      {
          p.direction = ts.p0 ;
          float r = radius  ;   

          float3 ringPosition = make_float3( r*cosPhi, r*sinPhi, 0.f ); 
          rotateUz(ringPosition, ts.p0);

          p.position = ts.x0 + ringPosition ;
          float3 photonPolarization = make_float3( sinPhi, -cosPhi, 0.f); // adhoc

          rotateUz(photonPolarization, ts.p0);

          p.polarization = photonPolarization ;
      }
      else if( ts.type == T_DISC_INTERSECT_SPHERE )
      {
          p.direction = ts.p0 ;

          float r = radius*sqrtf(u1) ;   

          float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f ); 

          p.position = ts.x0 + discPosition ;

          p.polarization = ts.mode & M_SPOL ? 
                                              normalize(make_float3(-discPosition.y, discPosition.x, 0.f)) 
                                            :
                                              normalize(make_float3(discPosition.x, discPosition.y, 0.f)) 
                                            ;  

      }
      else if( ts.type == T_DISC_INTERSECT_SPHERE_DUMB )
      {
          p.direction = ts.p0 ;

          float r = radius*sqrtf(u1) ;   

          float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f ); 
          rotateUz(discPosition, ts.p0);

          p.position = ts.x0 + discPosition ;

          {
              // look ahead sphere intersection in order to determine
              // surface normal at intersection so can set appropriate
              // polarization 

              float3 ray_origin = p.position ; 
              float3 ray_direction = make_float3( 0.f, 0.f, -1.f ) ; // -Z
              float3 sphere_center = make_float3( 0.f, 0.f, 0.f ) ;           
              float  sphere_radius = 100.f ; 

              float3 O = ray_origin - sphere_center   ; 
              float3 D = ray_direction ;
              
              float b = dot(O, D); 
              float c = dot(O, O) - sphere_radius*sphere_radius ; 
              float disc = b*b-c  ;

              if(disc > 0.f )
              {
                  float sdisc = sqrtf(disc) ;               
                  float root1 = (-b - sdisc);
                  float3 surfaceNormal = (O + root1*D)/sphere_radius;
                  p.polarization = ts.mode & M_SPOL ? 
                                                      normalize(cross(p.direction, surfaceNormal)) 
                                                    : 
                                                      p.direction   // TODO: PPOL should be transverse
                                                    ;  
              }
              else
              {
                  p.polarization = p.direction ; //TODO: should be transverse 
              }
          }

      }
      else if( ts.type == T_POINT )
      {
          p.direction = ts.p0 ;
          p.position = ts.x0 ;
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
          //  **REFLTEST**
          //
          //        *source* argument (ts.x0) is the center of the sphere 
          //        towards which the photons are directed
          //
          //        *target* argument is ignored, as direction comes from
          //        the inverse sphere position
          //
          //        *polarization* argument is used to set the surface normal vector
          //
          //        *mode* argument spol/ppol controls the initial polarization
          //
          //        Very special spherical geometry allows S-pol/P-pol direction vectors
          //        to be ontained by inspection:
          // 
          //           S-polarized : ie perpendicular to plane of incidence
          //           P-polarized : parallel to plane of incidence
          //
          //        Comparison of simulation results against analytic Fresnel formula 
          //        is simplified by use of M_FLAT_THETA to produce an initial 
          //        uniform distribution of incident angle.  Note the distribution is then 
          //        not uniform on area of the sphere, being bunched at poles. 
          //
          //        http://mathproofs.blogspot.tw/2005/04/uniform-random-distribution-on-sphere.html
          //
        
          float sinTheta, cosTheta;
          if(ts.mode & M_FLAT_COSTHETA )   
          { 
              cosTheta = 1.f - 2.0f*u1 ; 
              sinTheta = sqrtf( 1.0f - cosTheta*cosTheta );
          }
          else if( ts.mode & M_FLAT_THETA )  
          {
              sincosf(1.f*M_PIf*u1,&sinTheta,&cosTheta);
          }

          float3 spherePosition = make_float3( sinTheta*cosPhi, sinTheta*sinPhi, cosTheta ); 

          p.position = ts.x0 + radius*spherePosition ;

          p.direction = -spherePosition  ;

          p.polarization = ts.mode & M_SPOL ? 
                                               make_float3(spherePosition.y, -spherePosition.x , 0.f ) 
                                            :  
                                               make_float3(-spherePosition.x, -spherePosition.y , 0.f ) 
                                            ;  

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
          float3 photonPolarization = ts.mode & M_SPOL ? normalize(cross(p.direction, surfaceNormal)) : p.direction ;  
          p.polarization = photonPolarization ;

          unsigned long long photon_id = launch_index.x ;  

          float tdelta = float(photon_id % 100)*0.03333f ; // ripple effect 300 mm/ns * 0.0333 ns = ~10 mm between ripples
          p.time = ts.t0 + tdelta ; 

          //p.wavelength = float(photon_id % 10)*70.f + 100.f ; // toothcomb wavelength distribution 
      }
}


