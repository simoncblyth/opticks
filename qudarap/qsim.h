#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSIM_METHOD __device__
#else
   #define QSIM_METHOD 
#endif 

#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "sflow.h"
#include "sqat4.h"
#include "sc4u.h"
#include "sevent.h"   // enum {XYZ .. 

#include "qevent.h"
#include "qgs.h"
#include "qprop.h"
#include "qcurand.h"
#include "qbnd.h"
#include "qstate.h"

/**
qsim.h : GPU side struct prepared CPU side by QSim.hh
========================================================

Canonical use is from CSGOptiX/OptiX7Test.cu:simulate 

The qsim.h instance is uploaded once only at CSGOptiX instanciation 
as this encompasses the physics not the event-by-event info.


This is aiming to replace the OptiX 6 context in a CUDA-centric way.

* qsim encompasses global info relevant to to all photons, meaning that any changes
  make to the qsim instance from single photon threads must be into thread-owned slots 
  to avoid interference 
 
* temporary working state local to each photon is currently being passed by reference args, 
  would be cleaner to use a collective state struct to hold this local structs 

**/

struct curandStateXORWOW ; 
template <typename T> struct qprop ; 


template <typename T>
struct qsim
{
    qevent*             evt ; 
    curandStateXORWOW*  rngstate ; 

    cudaTextureObject_t scint_tex ; 
    quad4*              scint_meta ;
    // hmm could encapsulate the above group into a qscint ? 
    // and follow a similar approach for qcerenkov 
    // ... hmm there is commonality between the icdf textures with hd_factor on top 
    // that needs to be profited from 

    static constexpr T one = T(1.) ;   

    // boundary tex is created/configured CPU side in QBnd
    cudaTextureObject_t boundary_tex ; 
    quad4*              boundary_meta ; 
    unsigned            boundary_tex_MaterialLine_Water ;
    unsigned            boundary_tex_MaterialLine_LS ; 
    // hmm could encapsulate the above group into a qbnd ?

    quad*               optical ;  
    qprop<T>*           prop ;  
    int                 pidx ;   // from PIDX envvar

    static constexpr float hc_eVnm = 1239.8418754200f ; // G4: h_Planck*c_light/(eV*nm) 
 

// TODO: get more methods to work on CPU as well as GPU for easier testing 

    QSIM_METHOD void    generate_photon_dummy(      quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD void    generate_photon_torch(      quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 

    QSIM_METHOD static float3 uniform_sphere(curandStateXORWOW& rng); 
    QSIM_METHOD static float3 uniform_sphere(const float u0, const float u1); 
    QSIM_METHOD static void   rotateUz(float3& d, const float3& u ); 

// TODO: many of the below could be static, and many can work on CPU so bring them up here 
//       NB must also move the implementation

#if defined(__CUDACC__) || defined(__CUDABE__)

    QSIM_METHOD float4  boundary_lookup( unsigned ix, unsigned iy ); 
    QSIM_METHOD float4  boundary_lookup( float nm, unsigned line, unsigned k ); 


    QSIM_METHOD float   scint_wavelength_hd0(curandStateXORWOW& rng);  
    QSIM_METHOD float   scint_wavelength_hd10(curandStateXORWOW& rng);
    QSIM_METHOD float   scint_wavelength_hd20(curandStateXORWOW& rng);
    QSIM_METHOD void    scint_dirpol(quad4& p, curandStateXORWOW& rng); 
    QSIM_METHOD void    reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng);
    QSIM_METHOD void    scint_photon( quad4& p, GS& g, curandStateXORWOW& rng);
    QSIM_METHOD void    scint_photon( quad4& p, curandStateXORWOW& rng);

    QSIM_METHOD void    cerenkov_fabricate_genstep(GS& g, bool energy_range );
    QSIM_METHOD float   cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng, const GS& g);
    QSIM_METHOD float   cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng) ; 
    QSIM_METHOD void    cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id = -1 ) ; 
    QSIM_METHOD void    cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id = -1 ) ; 
    QSIM_METHOD void    cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id = -1 ) ; 
    QSIM_METHOD void    cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id = -1 ) ; 
    QSIM_METHOD void    cerenkov_photon_expt(  quad4& p, unsigned id, curandStateXORWOW& rng, int print_id = -1 ); 


    QSIM_METHOD void    generate_photon_carrier(    quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD void    generate_photon_simtrace(   quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD void    generate_photon(            quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 


    QSIM_METHOD void    fill_state(qstate& s, unsigned boundary, float wavelength, float cosTheta, unsigned idx ); 

    QSIM_METHOD static void lambertian_direction(float3* dir, const float3* normal, float orient, curandStateXORWOW& rng, unsigned idx  ); 
    QSIM_METHOD static void random_direction_marsaglia(float3* dir, curandStateXORWOW& rng, unsigned idx); 

    QSIM_METHOD static void   rayleigh_scatter_align(quad4& p, curandStateXORWOW& rng ); 


    QSIM_METHOD void    mock_propagate( quad4& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx ); 

    QSIM_METHOD int     propagate(const int bounce, quad4& p, qstate& s, const quad2* prd, curandStateXORWOW& rng, unsigned idx ); 
    QSIM_METHOD int     propagate_to_boundary(unsigned& flag, quad4& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx); 
    QSIM_METHOD int     propagate_at_surface( unsigned& flag, quad4& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx); 
    QSIM_METHOD int     propagate_at_boundary(unsigned& flag, quad4& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx); 

    QSIM_METHOD void    reflect_diffuse(  quad4& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx );
    QSIM_METHOD void    reflect_specular( quad4& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx );

    QSIM_METHOD void    hemisphere_polarized(   quad4& p, unsigned polz, bool inwards, const quad2* prd, curandStateXORWOW& rng); 


#else
    // instanciated on CPU and copied to device so no ctor in device code
    qsim()
        :
        evt(nullptr),
        rngstate(nullptr),
        scint_tex(0),
        scint_meta(nullptr),
        boundary_tex(0),
        boundary_meta(nullptr),
        optical(nullptr),
        prop(nullptr)
    {
    }
#endif

}; 




template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_dummy(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    printf("//qsim::generate_photon_dummy  photon_id %3d genstep_id %3d  gs.q0.i ( gencode:%3d %3d %3d %3d ) \n", 
       photon_id, 
       genstep_id, 
       gs.q0.i.x, 
       gs.q0.i.y,
       gs.q0.i.z, 
       gs.q0.i.w
      );  

    p.q0.i.x = 1 ; p.q0.i.y = 2 ; p.q0.i.z = 3 ; p.q0.i.w = 4 ; 
    p.q1.i.x = 1 ; p.q1.i.y = 2 ; p.q1.i.z = 3 ; p.q1.i.w = 4 ; 
    p.q2.i.x = 1 ; p.q2.i.y = 2 ; p.q2.i.z = 3 ; p.q2.i.w = 4 ; 
    p.q3.i.x = 1 ; p.q3.i.y = 2 ; p.q3.i.z = 3 ; p.q3.i.w = 4 ; 

    p.set_flag(TORCH); 
}




template <typename T>
inline QSIM_METHOD float3 qsim<T>::uniform_sphere(curandStateXORWOW& rng)
{
    float phi = curand_uniform(&rng)*2.f*M_PIf;
    float cosTheta = 2.f*curand_uniform(&rng) - 1.f ; // -1.f -> 1.f 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);
    return make_float3(cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta); 
}

template <typename T>
inline QSIM_METHOD float3 qsim<T>::uniform_sphere(const float u0, const float u1)
{
    float phi = u0*2.f*M_PIf;
    float cosTheta = 2.f*u1 - 1.f ; // -1.f -> 1.f 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);
    return make_float3(cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta); 
}

/**
qsim::rotateUz
---------------

This rotates the reference frame of a vector such that the original Z-axis will lie in the
direction of *u*. Many rotations would accomplish this; the one selected
uses *u* as its third column and is given by the below matrix.

The below CUDA implementation follows the CLHEP implementation used by Geant4::

     // geant4.10.00.p01/source/externals/clhep/src/ThreeVector.cc
     72 Hep3Vector & Hep3Vector::rotateUz(const Hep3Vector& NewUzVector) {
     73   // NewUzVector must be normalized !
     74 
     75   double u1 = NewUzVector.x();
     76   double u2 = NewUzVector.y();
     77   double u3 = NewUzVector.z();
     78   double up = u1*u1 + u2*u2;
     79 
     80   if (up>0) {
     81       up = std::sqrt(up);
     82       double px = dx,  py = dy,  pz = dz;
     83       dx = (u1*u3*px - u2*py)/up + u1*pz;
     84       dy = (u2*u3*px + u1*py)/up + u2*pz;
     85       dz =    -up*px +             u3*pz;
     86     }
     87   else if (u3 < 0.) { dx = -dx; dz = -dz; }      // phi=0  teta=pi
     88   else {};
     89   return *this;
     90 }

This implements rotation of (px,py,pz) vector into (dx,dy,dz) 
using the below rotation matrix, the columns of which must be 
orthogonal unit vectors.::

                |  u.x * u.z / up   -u.y / up    u.x  |        
        d  =    |  u.y * u.z / up   +u.x / up    u.y  |      p
                |   -up               0.         u.z  |      
    
Taking dot products between and within columns shows that to 
be the case for normalized u. See oxrap/rotateUz.h for the algebra. 

Special cases:

u = [0,0,1] (up=0.) 
   does nothing, effectively identity matrix

u = [0,0,-1] (up=0., u.z<0. ) 
   flip x, and z which is a rotation of pi/2 about y 

               |   -1    0     0   |
      d =      |    0    1     0   |   p
               |    0    0    -1   |
           
**/

template <typename T>
inline QSIM_METHOD void qsim<T>::rotateUz(float3& d, const float3& u ) 
{
    float up = u.x*u.x + u.y*u.y ;
    if (up>0.f) 
    {   
        up = sqrt(up);
        float px = d.x ;
        float py = d.y ;
        float pz = d.z ;
        d.x = (u.x*u.z*px - u.y*py)/up + u.x*pz;
        d.y = (u.y*u.z*px + u.x*py)/up + u.y*pz;
        d.z =    -up*px +                u.z*pz;
    }   
    else if (u.z < 0.f ) 
    {   
        d.x = -d.x; 
        d.z = -d.z; 
    }      
}



// TODO: get more of the below to work on CPU with mocked curand (and in future mocked tex2D and cudaTextureObject_t )

#if defined(__CUDACC__) || defined(__CUDABE__)

/**
qsim::boundary_lookup ix iy : Low level integer addressing lookup
--------------------------------------------------------------------

**/

template <typename T>
inline QSIM_METHOD float4 qsim<T>::boundary_lookup( unsigned ix, unsigned iy )
{
    const unsigned& nx = boundary_meta->q0.u.x  ; 
    const unsigned& ny = boundary_meta->q0.u.y  ; 
    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;
    float4 props = tex2D<float4>( boundary_tex, x, y );     
    return props ; 
}





/**
qsim::boundary_lookup nm line k 
----------------------------------

nm:    float wavelength 
line:  4*boundary_index + OMAT/OSUR/ISUR/IMAT   (0/1/2/3)
k   :  property group index 0/1 

return float4 props 

boundary_meta is required to configure access to the texture, 
it is uploaded by QTex::uploadMeta but requires calls to 


QTex::init automatically sets these from tex dimensions

   q0.u.x : nx width  (eg wavelength dimension)
   q0.u.y : ny hright (eg line dimension)

QTex::setMetaDomainX::

   q1.f.x : nm0  wavelength minimum in nm 
   q1.f.y : -
   q1.f.z : nms  wavelength step size in nm   

QTex::setMetaDomainY::

   q2.f.x : 
   q2.f.y :
   q2.f.z :





**/
template <typename T>
inline QSIM_METHOD float4 qsim<T>::boundary_lookup( float nm, unsigned line, unsigned k )
{
    //printf("//qsim.boundary_lookup nm %10.4f line %d k %d boundary_meta %p  \n", nm, line, k, boundary_meta  ); 

    const unsigned& nx = boundary_meta->q0.u.x  ; 
    const unsigned& ny = boundary_meta->q0.u.y  ; 
    const float& nm0 = boundary_meta->q1.f.x ; 
    const float& nms = boundary_meta->q1.f.z ; 

    float fx = (nm - nm0)/nms ;  
    float x = (fx+0.5f)/float(nx) ;   // ?? +0.5f ??

    unsigned iy = _BOUNDARY_NUM_FLOAT4*line + k ;   
    float y = (float(iy)+0.5f)/float(ny) ; 


    float4 props = tex2D<float4>( boundary_tex, x, y );     

    // printf("//qsim.boundary_lookup nm %10.4f nm0 %10.4f nms %10.4f  x %10.4f nx %d ny %d y %10.4f props.x %10.4f %10.4f %10.4f %10.4f  \n",
    //     nm, nm0, nms, x, nx, ny, y, props.x, props.y, props.z, props.w ); 

    return props ; 
}

/**
qsim::fill_state
-------------------

Formerly signed the 1-based boundary, now just keeping separate cosTheta to 
orient the use of the boundary so are using 0-based boundary. 

cosTheta < 0.f 
   photon direction is against the surface normal, ie are entering the shape
   
   * formerly this corresponded to -ve boundary 
   * line+OSUR is relevant surface
   * line+OMAT is relevant first material

cosTheta > 0.f 
   photon direction is with the surface normal, ie are exiting the shape
   
   * formerly this corresponded to +ve boundary
   * line+ISUR is relevant surface
   * line+IMAT is relevant first material


NB the line is above the details of the payload (ie how many float4 per matsur) it is just::
 
    boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 


The optical buffer is 4 times the length of the bnd, which allows
convenient access to the material and surface indices starting 
from a texture line.  See notes in:: 

    GBndLib::createOpticalBuffer 
    GBndLib::getOpticalBuf

Notice that s.optical.x and s.index.z are the same thing. 
So half of s.index is extraneous and the m1 index and m2 index 
is not much used.  

Also only one elemnt of m1group2 is actually used 


s.optical.x 
    used to distinguish between : boundary, surface (and in future multifilm)
    
    * currently contains 1-based surface index with 0 meaning "boundary" and anything else "surface"

    * TODO: encode boundary type enum into the high bits of s.optical.x for three way split 
      (perhaps use trigger strings like MULTIFILM in the boundary spec to configure)
      THIS WILL NEED TO BE DONE AT x4 translation level (and repeated in QBndOptical for 
      dynamic boundary adding) 

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::fill_state(qstate& s, unsigned boundary, float wavelength, float cosTheta, unsigned idx  )
{
    const int line = boundary*_BOUNDARY_NUM_MATSUR ;      // now that are not signing boundary use 0-based

    const int m1_line = cosTheta > 0.f ? line + IMAT : line + OMAT ;   
    const int m2_line = cosTheta > 0.f ? line + OMAT : line + IMAT ;   
    const int su_line = cosTheta > 0.f ? line + ISUR : line + OSUR ;   


    s.material1 = boundary_lookup( wavelength, m1_line, 0);   // refractive_index, absorption_length, scattering_length, reemission_prob
    s.m1group2  = boundary_lookup( wavelength, m1_line, 1);   // group_velocity ,  (unused          , unused           , unused)  
    s.material2 = boundary_lookup( wavelength, m2_line, 0);   // refractive_index, (absorption_length, scattering_length, reemission_prob) only m2:refractive index actually used  
    s.surface   = boundary_lookup( wavelength, su_line, 0);   //  detect,        , absorb            , (reflect_specular), reflect_diffuse     [they add to 1. so one not used] 

    //printf("//qsim.fill_state boundary %d line %d wavelength %10.4f m1_line %d \n", boundary, line, wavelength, m1_line ); 

    s.optical = optical[su_line].u ;   // 1-based-surface-index-0-meaning-boundary/type/finish/value  (type,finish,value not used currently)

    //printf("//qsim.fill_state idx %d boundary %d line %d wavelength %10.4f m1_line %d m2_line %d su_line %d s.optical.x %d \n", 
    //    idx, boundary, line, wavelength, m1_line, m2_line, su_line, s.optical.x ); 

    s.index.x = optical[m1_line].u.x ; // m1 index
    s.index.y = optical[m2_line].u.x ; // m2 index 
    s.index.z = optical[su_line].u.x ; // su index
    s.index.w = 0u ;                   // avoid undefined memory comparison issues

    //printf("//qsim.fill_state \n"); 
}


/**
qsim::lambertian_direction following G4LambertianRand 
--------------------------------------------------------

g4-cls G4RandomTools::

     59 inline G4ThreeVector G4LambertianRand(const G4ThreeVector& normal)
     60 {
     61   G4ThreeVector vect;
     62   G4double ndotv;
     63   G4int count=0;
     64   const G4int max_trials = 1024;
     65 
     66   do
     67   {
     68     ++count;
     69     vect = G4RandomDirection();
     70     ndotv = normal * vect;
     71 
     72     if (ndotv < 0.0)
     73     {
     74       vect = -vect;
     75       ndotv = -ndotv;
     76     }
     77 
     78   } while (!(G4UniformRand() < ndotv) && (count < max_trials));
     79 
     80   return vect;
     81 }


NB: potentially bad for performance for dir pointer to be into global mem
as opposed to local stack float3 : as this keeps changing the dir before 
arriving at the final one

**/
template <typename T>
inline  QSIM_METHOD void qsim<T>::lambertian_direction(float3* dir, const float3* normal, float orient, curandStateXORWOW& rng, unsigned idx )
{
    float ndotv ; 
    int count = 0 ; 
    float u ; 
    do
    {
        count++ ; 
        random_direction_marsaglia(dir, rng, idx); 
        ndotv = dot( *dir, *normal )*orient ; 
        if( ndotv < 0.f )
        {
            *dir = -1.f*(*dir) ; 
            ndotv = -1.f*ndotv ; 
        } 
        u = curand_uniform(&rng) ; 

        //if( idx == 0u) printf("//qsim.lambertian_direction idx %d count %d  u %10.4f \n", idx, count, u ); 

    } 
    while (!(u < ndotv) && (count < 1024)) ;  
}


/**
qsim::random_direction_marsaglia following G4RandomDirection
-------------------------------------------------------------

* https://mathworld.wolfram.com/SpherePointPicking.html

Marsaglia (1972) derived an elegant method that consists of picking u and v from independent 
uniform distributions on (-1,1) and rejecting points for which uu+vv >=1. 
From the remaining points,

    x=2 u sqrt(1-(uu+vv))	
    y=2 v sqrt(1-(uu+vv))	
    z=1-2(uu+vv)

Checking normalization, it reduces to 1::

   xx + yy + zz = 
         4uu (1-(uu+vv)) 
         4vv (1-(uu+vv)) +
        1 -4(uu+vv) + 4(uu+vv)(uu+vv)   
                = 1 

::

                          v
                          |
              +---------.-|- -----------+
              |      .    |     .       |
              |   .       |          .  |
              |           |             |
              | .         |            .|
              |           |             |
          ----+-----------0-------------+---- u
              |.          |             |
              |           |            .|
              | .         |             |
              |           |          .  |
              |    .      |      .      |
              +--------.--|--.----------+
                          |      

::

     g4-cls G4RandomDirection

     58 // G.Marsaglia (1972) method
     59 inline G4ThreeVector G4RandomDirection()
     60 {
     61   G4double u, v, b;
     62   do {
     63     u = 2.*G4UniformRand() - 1.;
     64     v = 2.*G4UniformRand() - 1.;
     65     b = u*u + v*v;
     66   } while (b > 1.);
     67   G4double a = 2.*std::sqrt(1. - b);
     68   return G4ThreeVector(a*u, a*v, 2.*b - 1.);
     69 }

**/


template <typename T>
inline QSIM_METHOD void qsim<T>::random_direction_marsaglia(float3* dir,  curandStateXORWOW& rng, unsigned idx )
{
    float u0, u1 ; 

    float u, v, b, a  ; 
    do 
    {
        u0 = curand_uniform(&rng);
        u1 = curand_uniform(&rng);
        //if( idx == 0u ) printf("//qsim.random_direction_marsaglia idx %d u0 %10.4f u1 %10.4f \n", idx, u0, u1 ); 

        u = 2.f*u0 - 1.f ; 
        v = 2.f*u1 - 1.f ; 
        b = u*u + v*v ; 
    } 
    while( b > 1.f ) ; 

    a = 2.f*sqrtf( 1.f - b );   

    dir->x = a*u ; 
    dir->y = a*v ; 
    dir->z = 2.f*b - 1.f ; 
}



/**
qsim::rayleigh_scatter_align
------------------------------

Following G4OpRayleigh::PostStepDoIt

* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=207 Xin Qian patch


Transverse wave nature means::

   dot(p_direction, p_polarization)  = 0 
   dot(direction,   polarization)  = 0 

*constant* and normalized direction retains transversality thru the candidate scatter::

    pol = p_pol + constant*dir

    dot(pol, dir) = dot(p_pol, dir) + constant* dot(dir, dir)
                  = dot(p_pol, dir) + constant* 1. 
                  = dot(p_pol, dir) - dot(p_pol, dir)
                  = 0.
      
**/

template <typename T>
inline QSIM_METHOD void qsim<T>::rayleigh_scatter_align(quad4& p, curandStateXORWOW& rng )
{
    float3* p_direction = (float3*)&p.q1.f.x ; 
    float3* p_polarization = (float3*)&p.q2.f.x ; 
    float3 direction ; 
    float3 polarization ; 

    bool looping(true) ;  
    do 
    {
        float u0 = curand_uniform(&rng) ;    
        float u1 = curand_uniform(&rng) ;    
        float u2 = curand_uniform(&rng) ;    
        float u3 = curand_uniform(&rng) ;    
        float u4 = curand_uniform(&rng) ;    

        float cosTheta = u0 ;
        float sinTheta = sqrtf(1.0f-u0*u0);
        if(u1 < 0.5f ) cosTheta = -cosTheta ; 
        // could use uniform_sphere here : but not doing so to closesly follow G4OpRayleigh

        float sinPhi ; 
        float cosPhi ; 
        sincosf(2.f*M_PIf*u2,&sinPhi,&cosPhi);
       
        direction.x = sinTheta * cosPhi;
        direction.y = sinTheta * sinPhi;
        direction.z = cosTheta ;

        rotateUz(direction, *p_direction );

        float constant = -dot(direction,*p_polarization); 

        polarization.x = p_polarization->x + constant*direction.x ;
        polarization.y = p_polarization->y + constant*direction.y ;
        polarization.z = p_polarization->z + constant*direction.z ;

        if(dot(polarization, polarization) == 0.f )
        {
            sincosf(2.f*M_PIf*u3,&sinPhi,&cosPhi);

            polarization.x = cosPhi ;
            polarization.y = sinPhi ;
            polarization.z = 0.f ;

            rotateUz(polarization, direction);
        }
        else
        {
            // There are two directions which are perpendicular
            // to the new momentum direction
            if(u3 < 0.5f) polarization = -polarization ;
        }
        polarization = normalize(polarization);

        // simulate according to the distribution cos^2(theta)
        // where theta is the angle between old and new polarizations
        float doCosTheta = dot(polarization,*p_polarization) ;
        float doCosTheta2 = doCosTheta*doCosTheta ;
        looping = doCosTheta2 < u4 ;

    } while ( looping ) ;

    *p_direction = direction ;
    *p_polarization = polarization ;
}


/**
qsim::propagate_to_boundary
------------------------------

+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
| flag                |   command        |  changed                                                |  note                                                 |
+=====================+==================+=========================================================+=======================================================+
|   BULK_REEMIT       |   CONTINUE       |  time, position, direction, polarization, wavelength    | advance to reemit position with everything changed    |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   BULK_SCATTER      |   CONTINUE       |  time, position, direction, polarization                | advance to scatter position, new dir+pol              |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   BULK_ABSORB       |   BREAK          |  time, position                                         | advance to absorption position, dir+pol unchanged     |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   not set "SAIL"    |   BOUNDARY       |  time, position                                         | advanced to border position, dir+pol unchanged        |   
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+


TODO: whilst in measurement iteration try changing to a single return, not loads of them, by setting command and returning that 

**/

template <typename T>
inline QSIM_METHOD int qsim<T>::propagate_to_boundary(unsigned& flag, quad4& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
{
    const float& absorption_length = s.material1.y ; 
    const float& scattering_length = s.material1.z ; 
    const float& reemission_prob = s.material1.w ; 
    const float& group_velocity = s.m1group2.x ; 
    const float& distance_to_boundary = prd->q0.f.w ; 



    float3* position = (float3*)&p.q0.f.x ; 
    float* time = &p.q0.f.w ;  
    float3* direction = (float3*)&p.q1.f.x ; 
    float3* polarization = (float3*)&p.q2.f.x ; 
    float* wavelength = &p.q2.f.w ; 
    //int4& flags = p.q3.i ;  

    float u_scattering = curand_uniform(&rng) ;
    float u_absorption = curand_uniform(&rng) ;
    float scattering_distance = -scattering_length*logf(u_scattering);   
    float absorption_distance = -absorption_length*logf(u_absorption);

#ifdef DEBUG_TIME
    if( idx == pidx ) printf("//qsim.propagate_to_boundary[ idx %d post (%10.4f %10.4f %10.4f %10.4f) \n", idx, position->x, position->y, position->z, *time );  
#endif

#ifdef DEBUG_HIST
    if(idx == pidx ) printf("//qsim.propagate_to_boundary idx %d distance_to_boundary %10.4f absorption_distance %10.4f scattering_distance %10.4f u_scattering %10.4f u_absorption %10.4f \n", 
             idx, distance_to_boundary, absorption_distance, scattering_distance, u_scattering, u_absorption  ); 
#endif
  

    if (absorption_distance <= scattering_distance) 
    {   
        if (absorption_distance <= distance_to_boundary) 
        {   
            *time += absorption_distance/group_velocity ;   
            *position += absorption_distance*(*direction) ;

#ifdef DEBUG_TIME
            float absorb_time_delta = absorption_distance/group_velocity ; 
            if( idx == pidx ) printf("//qsim.propagate_to_boundary] idx %d post (%10.4f %10.4f %10.4f %10.4f) absorb_time_delta %10.4f   \n", 
                         idx, position->x, position->y, position->z, *time, absorb_time_delta  );  
#endif

            float u_reemit = reemission_prob == 0.f ? 2.f : curand_uniform(&rng);  // avoid consumption at absorption when not scintillator

            if (u_reemit < reemission_prob)    
            {   
                *wavelength = scint_wavelength_hd20(rng);
                *direction = uniform_sphere(rng);
                *polarization = normalize(cross(uniform_sphere(rng), *direction));

                flag = BULK_REEMIT ;
                return CONTINUE;
            }    
            else 
            {   
                flag = BULK_ABSORB ;
                return BREAK;
            }    
        }   
        //  otherwise sail to boundary  
    }   
    else 
    {   
        if (scattering_distance <= distance_to_boundary)
        {
            *time += scattering_distance/group_velocity ;
            *position += scattering_distance*(*direction) ;

            rayleigh_scatter_align(p, rng); // changes dir and pol, consumes 5u at each turn of rejection sampling loop

            flag = BULK_SCATTER;

            return CONTINUE;
        }       
          //  otherwise sail to boundary  
    }     // if scattering_distance < absorption_distance



    *position += distance_to_boundary*(*direction) ;
    *time     += distance_to_boundary/group_velocity   ;  

#ifdef DEBUG_TIME
    float sail_time_delta = distance_to_boundary/group_velocity ; 
    if( idx == pidx ) printf("//qsim.propagate_to_boundary] idx %d post (%10.4f %10.4f %10.4f %10.4f) sail_time_delta %10.4f   \n", 
          idx, position->x, position->y, position->z, *time, sail_time_delta  );  
#endif

    return BOUNDARY ;
}

/**
qsim::propagate_at_boundary
------------------------------------------

This was brought over from oxrap/cu/propagate.h:propagate_at_boundary_geant4_style 
See env-/g4op-/G4OpBoundaryProcess.cc annotations to follow this
and compare the Opticks and Geant4 implementations.

Input:

* p.direction
* p.polarization
* s.material1.x    : refractive index 
* s.material2.x    : refractive index
* prd.normal 

Consumes one random deciding between BOUNDARY_REFLECT and BOUNDARY_TRANSMIT

+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
| output flag         |   command        |  changed                                                |  note                                                 |
+=====================+==================+=========================================================+=======================================================+
|   BOUNDARY_REFLECT  |    -             |  direction, polarization                                |                                                       |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   BOUNDARY_TRANSMIT |    -             |  direction, polarization                                |                                                       |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+

Notes:

* when geometry and refractive indices dictates TIR there is no dependence on u_reflect and always get reflection


::

                    s1
                  +----+          
                   \   .   /      ^
              c1   i\  .  / r    /|\
                     \ . /        |                      
        material1     \./         | n
        ---------------+----------+----------
        material2      .\
                       . \
                  c2   .  \ t
                       .   \
                       +----+
                         s2


Snells law::

     s1    n2 
    --- = ---         s1 n1 = s2 n2         eta = n1/n2     s1 eta = s2 
     s2    n1

     s1.s1 = 1 - c1.c1   # trig identity
 
     s2.s2 = 1 - c2.c2

    s1 eta = s2          # snell 

    s1s1 eta eta = s2s2 

    ( 1.f - c1c1 ) eta eta = 1.f - c2c2
 
     c2c2 = 1.f - eta eta ( 1.f - c1c1 )    # snell and trig identity 

Because the electromagnetic wave is transverse, the field incident onto the
interface can be decomposed into two polarization components, one P-polarized,
i.e., with the electric field vector inside the plane of incidence, and the
other one S-polarized, i.e., orthogonal to that plane.


inconsistent normal definitions, c1 is expected to be +ve and normal needs to be oriented against initial direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is apparent from reflected direction vector::

      *direction + 2.0f*c1*surface_normal


The standard normal vector at an intersection position on the surface of a shape 
is defined to be rigidly oriented outwards away from the shape.  
This definition is used by *fill_state* to order determine proparties 
if this material m1 and the next material m2 on the other side of the boundary.   

The below math assumes that the photon direction is always against the normal 
such that the sign of c1 is +ve. Having -ve c1 leads to non-sensical -ve TranCoeff
which results in always relecting. 

So what about photons going in the other direction ? 
Surface normal is used in several places in the below so presumably must 
arrange to have an oriented normal that is flipped appropriately OR perhaps can change the math ?

In summary this is a case of inconsistent definitions of the normal, 
that will need to be oriented ~half the time. 

TODO: try avoiding "float3 oriented_normal" instead just use "bool orient" 
      and multiply prd.normal by 1.f or -1.f depending on orient at every use


random aligned matching with examples/Geant/BoundaryStandalone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* S/P/"X"-polarized + TIR + normal_incidence all now matching
* noted a 1-in-a-million deviant from TransCoeff cut edge float/double for S and P

* initially had two more deviants at very close to normal incidence that were aligned by changing 
  the criteria to match Geant4 "sint1 == 0." better::

    //const bool normal_incidence = fabs(c1) > 0.999999f ; 
    const bool normal_incidence = fabs(c1) == 1.f ; 

* see notes/issues/QSimTest_propagate_at_boundary_vs_BoundaryStandalone_G4OpBoundaryProcessTest.rst

**/

template <typename T>
inline QSIM_METHOD int qsim<T>::propagate_at_boundary(unsigned& flag, quad4& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
{
    const float& n1 = s.material1.x ;
    const float& n2 = s.material2.x ;   
    const float eta = n1/n2 ; 

    float3* direction    = (float3*)&p.q1.f.x ; 
    float3* polarization = (float3*)&p.q2.f.x ; 
    const float3* normal = (float3*)&prd->q0.f.x ; 

    const float _c1 = -dot(*direction, *normal ); 
    const float3 oriented_normal = _c1 < 0.f ? -(*normal) : (*normal) ; 
    const float c1 = fabs(_c1) ; 
    const bool normal_incidence = c1 == 1.f ; 

    /* 
    printf("//qsim.propagate_at_boundary idx %d nrm   (%10.4f %10.4f %10.4f) \n", idx, oriented_normal.x, oriented_normal.y, oriented_normal.z ); 
    printf("//qsim.propagate_at_boundary idx %d mom_0 (%10.4f %10.4f %10.4f) \n", idx, direction->x, direction->y, direction->z ); 
    printf("//qsim.propagate_at_boundary idx %d pol_0 (%10.4f %10.4f %10.4f) \n", idx, polarization->x, polarization->y, polarization->z ); 
    printf("//qsim.propagate_at_boundary idx %d c1 %10.4f normal_incidence %d \n", idx, c1, normal_incidence ); 
    */

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law and trig identity 
    bool tir = c2c2 < 0.f ; 
    const float EdotN = dot(*polarization, oriented_normal ) ;  // used for TIR polarization
    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect
    const float n1c1 = n1*c1 ;
    const float n2c2 = n2*c2 ; 
    const float n2c1 = n2*c1 ; 
    const float n1c2 = n1*c2 ; 
    const float3 A_trans = normal_incidence ? *polarization : normalize(cross(*direction, oriented_normal)) ; // perpendicular to plane of incidence
    const float E1_perp = dot(*polarization, A_trans);     //  E vector component perpendicular to plane of incidence, ie S polarization
    const float2 E1   = normal_incidence ? make_float2( 0.f, 1.f) : make_float2( E1_perp , length( *polarization - (E1_perp*A_trans) ) ); 
    const float2 E2_t = make_float2(  2.f*n1c1*E1.x/(n1c1+n2c2), 2.f*n1c1*E1.y/(n2c1+n1c2) ) ;  // ( S:perp, P:parl )  
    const float2 E2_r = make_float2( E2_t.x - E1.x             , (n2*E2_t.y/n1) - E1.y     ) ;  // ( S:perp, P:parl )    
    const float2 RR = normalize(E2_r) ; 
    const float2 TT = normalize(E2_t) ; 
    const float TransCoeff = tir || n1c1 == 0.f ? 0.f : n2c2*dot(E2_t,E2_t)/n1c1 ; 

    const float u_boundary_burn = curand_uniform(&rng) ;  // needed for random consumption alignment with Geant4 G4OpBoundaryProcess::PostStepDoIt
    const float u_reflect = curand_uniform(&rng) ;
    bool reflect = u_reflect > TransCoeff  ;

    //printf("//qsim.propagate_at_boundary n2c2 %10.4f n1c1 %10.4f u_reflect %10.4f TransCoeff %10.4f (n2c2.E2_total_t/n1c1)  reflect %d \n", 
    //                                          n2c2,  n1c1, u_reflect, TransCoeff, reflect ); 

    // dirty debug stomping on 
    //p.q0.f.w = u_reflect ;   // non-standard 
    //p.q1.f.w = TransCoeff ;  // non-standard replace "weight"

    /*
    if(idx == 251959)
    {
        printf("//qsim.propagate_at_boundary idx %d \n", idx); 
        printf("//qsim.propagate_at_boundary oriented_normal (%10.4f, %10.4f, %10.4f) \n", oriented_normal.x, oriented_normal.y, oriented_normal.z );  
        printf("//qsim.propagate_at_boundary direction (%10.4f, %10.4f, %10.4f) \n", direction->x, direction->y, direction->z );  
        printf("//qsim.propagate_at_boundary polarization (%10.4f, %10.4f, %10.4f) \n", polarization->x, polarization->y, polarization->z );  
        printf("//qsim.propagate_at_boundary c1 %10.4f normal_incidence %d \n", c1, normal_incidence ); 
    }
   */


    *direction = reflect
                    ?
                       *direction + 2.0f*c1*oriented_normal
                    :
                       eta*(*direction) + (eta*c1 - c2)*oriented_normal
                    ;


    const float3 A_paral = normalize(cross(*direction, A_trans));

    *polarization =  normal_incidence ?
                                         ( reflect ?  *polarization*(n2>n1? -1.f:1.f) : *polarization )
                                      : 
                                         ( reflect ?
                                                   ( tir ?  -(*polarization) + 2.f*EdotN*oriented_normal : RR.x*A_trans + RR.y*A_paral )

                                                   :
                                                       TT.x*A_trans + TT.y*A_paral 
                                             
                                                   )
                                      ;

    /*
    printf("//qsim.propagate_at_boundary idx %d reflect %d tir %d TransCoeff %10.4f u_reflect %10.4f \n", idx, reflect, tir, TransCoeff, u_reflect );  
    printf("//qsim.propagate_at_boundary idx %d mom_1 (%10.4f %10.4f %10.4f) \n", idx, direction->x, direction->y, direction->z ); 
    printf("//qsim.propagate_at_boundary idx %d pol_1 (%10.4f %10.4f %10.4f) \n", idx, polarization->x, polarization->y, polarization->z ); 
    */

    /*
    if(idx == 251959)
    {
        printf("//qsim.propagate_at_boundary RR.x %10.4f A_trans (%10.4f %10.4f %10.4f )  RR.y %10.4f  A_paral (%10.4f %10.4f %10.4f ) \n", 
              RR.x, A_trans.x, A_trans.y, A_trans.z,
              RR.y, A_paral.x, A_paral.y, A_paral.z ); 

        printf("//qsim.propagate_at_boundary reflect %d  tir %d polarization (%10.4f, %10.4f, %10.4f) \n", reflect, tir, polarization->x, polarization->y, polarization->z );  
    }
    */

    flag = reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ; 
    return CONTINUE ; 
}

/*
G4OpBoundaryProcess::DielectricDielectric


1152               if (sint1 > 0.0) {
1153                  A_trans = OldMomentum.cross(theFacetNormal);
1154                  A_trans = A_trans.unit();
1155                  E1_perp = OldPolarization * A_trans;
1156                  E1pp    = E1_perp * A_trans;
1157                  E1pl    = OldPolarization - E1pp;
1158                  E1_parl = E1pl.mag();
1159               }
1160               else {
1161                  A_trans  = OldPolarization;
1162                  // Here we Follow Jackson's conventions and we set the
1163                  // parallel component = 1 in case of a ray perpendicular
1164                  // to the surface
1165                  E1_perp  = 0.0;
1166                  E1_parl  = 1.0;
1167               }
1168 
1169               s1 = Rindex1*cost1;
1170               E2_perp = 2.*s1*E1_perp/(Rindex1*cost1+Rindex2*cost2);
1171               E2_parl = 2.*s1*E1_parl/(Rindex2*cost1+Rindex1*cost2);
1172               E2_total = E2_perp*E2_perp + E2_parl*E2_parl;
1173               s2 = Rindex2*cost2*E2_total;
1174 
1175               if (theTransmittance > 0) TransCoeff = theTransmittance;
1176               else if (cost1 != 0.0) TransCoeff = s2/s1;
1177               else TransCoeff = 0.0;


reflect

1217                     
1218                        E2_parl   = Rindex2*E2_parl/Rindex1 - E1_parl;
1219                        E2_perp   = E2_perp - E1_perp;
1220                        E2_total  = E2_perp*E2_perp + E2_parl*E2_parl;
1221                        A_paral   = NewMomentum.cross(A_trans);
1222                        A_paral   = A_paral.unit();
1223                        E2_abs    = std::sqrt(E2_total);
1224                        C_parl    = E2_parl/E2_abs;
1225                        C_perp    = E2_perp/E2_abs;
1226 
1227                        NewPolarization = C_parl*A_paral + C_perp*A_trans;
1228 

transmit 

1253                    alpha = cost1 - cost2*(Rindex2/Rindex1);
1254                    NewMomentum = OldMomentum + alpha*theFacetNormal;
1255                    NewMomentum = NewMomentum.unit();
1256 //                   PdotN = -cost2;
1257                    A_paral = NewMomentum.cross(A_trans);
1258                    A_paral = A_paral.unit();

1259                    E2_abs     = std::sqrt(E2_total);
1260                    C_parl     = E2_parl/E2_abs;
1261                    C_perp     = E2_perp/E2_abs;
1262 
1263                    NewPolarization = C_parl*A_paral + C_perp*A_trans;

*/


/**
qsim::propagate_at_surface
----------------------------

+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
| output flag         |   command        |  changed                                                |  note                                                 |
+=====================+==================+=========================================================+=======================================================+
|   SURFACE_ABSORB    |    BREAK         |                                                         |                                                       |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   SURFACE_DETECT    |    BREAK         |                                                         |                                                       |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   SURFACE_DREFLECT  |    CONTINUE      |   direction, polarization                               |                                                       |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+
|   SURFACE_SREFLECT  |    CONTINUE      |   direction, polarization                               |                                                       |
+---------------------+------------------+---------------------------------------------------------+-------------------------------------------------------+

**/

template <typename T>
inline QSIM_METHOD int qsim<T>::propagate_at_surface(unsigned& flag, quad4& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
{
    const float& detect = s.surface.x ;
    const float& absorb = s.surface.y ;
    //const float& reflect_specular_ = s.surface.z ; 
    const float& reflect_diffuse_  = s.surface.w ; 

    float u_surface = curand_uniform(&rng);
    float u_surface_burn = curand_uniform(&rng);

    int action = u_surface < absorb + detect ? BREAK : CONTINUE  ; 

    if( action == BREAK )
    {
        flag = u_surface < absorb ? SURFACE_ABSORB : SURFACE_DETECT  ;
    }
    else 
    {
        flag = u_surface < absorb + detect + reflect_diffuse_ ?  SURFACE_DREFLECT : SURFACE_SREFLECT ;  
        switch(flag)
        {
            case SURFACE_DREFLECT: reflect_diffuse( p, prd, rng, idx)  ; break ; 
            case SURFACE_SREFLECT: reflect_specular(p, prd, rng, idx)  ; break ; 
        }
    }
    return action ; 
}


/**
qsim::reflect_diffuse cf G4OpBoundaryProcess::DoReflection
-----------------------------------------------------------

* LobeReflection is not yet implemnented in qsim.h 

::

    355 inline
    356 void G4OpBoundaryProcess_MOCK::DoReflection()
    357 {
    358         if ( theStatus == LambertianReflection ) {
    359 
    360           NewMomentum = G4LambertianRand(theGlobalNormal);
    361           theFacetNormal = (NewMomentum - OldMomentum).unit();
    362 
    363         }
    364         else if ( theFinish == ground ) {
    365 
    366           theStatus = LobeReflection;
    367           if ( PropertyPointer1 && PropertyPointer2 ){
    368           } else {
    369              theFacetNormal =
    370                  GetFacetNormal(OldMomentum,theGlobalNormal);
    371           }
    372           G4double PdotN = OldMomentum * theFacetNormal;
    373           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    374 
    375         }
    376         else {
    377 
    378           theStatus = SpikeReflection;
    379           theFacetNormal = theGlobalNormal;
    380           G4double PdotN = OldMomentum * theFacetNormal;
    381           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    382 
    383         }
    384         G4double EdotN = OldPolarization * theFacetNormal;
    385         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    386 }


**/

template <typename T>
inline QSIM_METHOD void qsim<T>::reflect_diffuse( quad4& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx )
{
    float3* dir = (float3*)&p.q1.f.x ;  
    float3* pol = (float3*)&p.q2.f.x ;  

    //if(idx == 0 ) printf("//qsim.reflect_diffuse idx %d dir0 (%10.4f %10.4f %10.4f) pol (%10.4f %10.4f %10.4f) \n", idx, dir->x, dir->y, dir->z, pol->x, pol->y, pol->z  ); 

    float3 old_dir = *dir ; 

    const float3* normal = prd->normal()  ;  
    const float orient = -1.f ;     // equivalent to G4OpBoundaryProcess::PostStepDoIt early flip  of theGlobalNormal ?
    lambertian_direction(dir, normal, orient, rng, idx );

    float3 facet_normal = normalize( *dir - old_dir ); 
    const float EdotN = dot( *pol, facet_normal ); 
    *pol = -1.f*(*pol) + 2.f*EdotN*facet_normal ; 
}

template <typename T>
inline QSIM_METHOD void qsim<T>::reflect_specular( quad4& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx )
{
    float3* dir = (float3*)&p.q1.f.x ;  
    float3* pol = (float3*)&p.q2.f.x ;  

    const float3* normal = prd->normal() ;      
    const float orient = -1.f ;     // equivalent to G4OpBoundaryProcess::PostStepDoIt early flip of theGlobalNormal ?

    const float PdotN = dot( *dir, *normal )*orient ; 
    *dir = *dir - 2.f*PdotN*(*normal)*orient ; 

    const float EdotN = dot( *pol, *normal )*orient ; 
    *pol = -1.f*(*pol) + 2.f*EdotN*(*normal)*orient  ; 
}

/**
qsim::mock_propagate
----------------------

TODO
~~~~~

* can qstate be slimmed : seems not very easily 
* simplify qstate persisting (quad6?quad5?)
* compressed step *seq* recording  


Stages within bounce loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. mock call to OptiX trace : doing "geometry lookup"
   photon position + direction -> surface normal + distance and identity, boundary 

   * cosTheta sign gives boundary orientation   

2. lookup material/surface properties using boundary and orientation (cosTheta) from geometry lookup 

3. mutate photon and set flag using material properties

  * note that photons that SAIL to boundary are mutated twice within the while loop (by propagate_to_boundary and propagate_at_boundary/surface) 


TODO: record and record_max should come from qevent ?

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::mock_propagate( quad4& p, const quad2* mock_prd, curandStateXORWOW& rng, unsigned idx )
{
    p.set_flag(TORCH);  // setting initial flag : in reality this should be done by generation

    quad4* record = evt->record ;  
    const int& max_record = evt->max_record ; 
    const int& max_bounce = evt->max_bounce ; 
    printf("//qsim.mock_propagate evt.max_bounce %d evt.max_record %d evt.record %p evt.num_record %d \n", max_bounce, max_record, record, evt->num_record ); 

    int bounce = 0 ; 
    int command = START ; 
    qstate s ; 
    while( bounce < max_bounce )
    {
        if(record) record[max_record*idx+bounce] = p ;  
        const quad2* prd = mock_prd + (max_bounce*idx+bounce) ;  
        command = propagate(bounce, p, s, prd, rng, idx ); 
        bounce++;        
        if(command == BREAK) break ;    
    }
    if( record && bounce < max_record ) record[max_record*idx+bounce] = p ;  
}

/**
qsim::propagate : one "bounce" propagate_to_boundary/propagate_at_boundary 
-----------------------------------------------------------------------------

This is canonically invoked from within the bounce loop of CSGOptiX/OptiX7Test.cu:simulate 

TODO: missing needs to return BREAK   

**/

template <typename T>
inline QSIM_METHOD int qsim<T>::propagate(const int bounce, quad4& p, qstate& s, const quad2* prd, curandStateXORWOW& rng, unsigned idx ) 
{
    float* wavelength = &p.q2.f.w ; 
    float3* dir = (float3*)&p.q1.f.x ;    

    const unsigned boundary = prd->boundary() ; 
    const unsigned identity = prd->identity() ; 
    const float3* normal = prd->normal(); 
    float cosTheta = dot(*dir, *normal ) ;    

#ifdef DEBUG_COSTHETA
    if( idx == pidx ) printf("//qsim.propagate idx %d bnc %d cosTheta %10.4f dir (%10.4f %10.4f %10.4f) nrm (%10.4f %10.4f %10.4f) \n", 
                 idx, bounce, cosTheta, dir->x, dir->y, dir->z, normal->x, normal->y, normal->z ); 
#endif

    p.set_prd(boundary, identity, cosTheta); 

    fill_state(s, boundary, *wavelength, cosTheta, idx ); 

    unsigned flag = 0 ;  

    int command = propagate_to_boundary( flag, p, prd, s, rng, idx ); 
    //if( idx == 0 ) 
    printf("//qsim.propagate idx %d bounce %d command %d flag %d s.optical.x %d \n", idx, bounce, command, flag, s.optical.x  );   

    if( command == BOUNDARY )
    {
        command = s.optical.x > 0 ? 
                                      propagate_at_surface( flag, p, prd, s, rng, idx ) 
                                  : 
                                      propagate_at_boundary( flag, p, prd, s, rng, idx) 
                                  ;  
    }

    p.set_flag(flag);    // hmm could hide this ?

    return command ; 
}






/**
qsim::hemisphere_s_polarized
------------------------------


          direction      | surface_normal
               \         |
                \        |
              .  \       |
          .       \      |
      .            \     | 
     within         \    |
                     \   |
                      \  |
                       \ |
                        \|
           --------------+------------
    

*plane of incidence*
    plane containing *surface_normal* *direction* and *within* vectors 
               
*transverse* 
    vector transverse to the plane of incidence (S polarized)
    
*within*
    vector within the plane of incidence and perpendicular to *direction* (P polarized)


A +ve Z upper hemisphere of *direction* is generated and then rotateUz oriented 
to adopt the *surface_normal* vector as its Z direction.

For inwards=true the normal direction is flipped to orient all the directions 
inwards. 
 
**/

template <typename T>
inline QSIM_METHOD void qsim<T>::hemisphere_polarized(quad4& p, unsigned polz, bool inwards, const quad2* prd, curandStateXORWOW& rng)
{
    const float3* normal = prd->normal() ; 

    float3* direction    = (float3*)&p.q1.f.x ; 
    float3* polarization = (float3*)&p.q2.f.x ; 

    //printf("//qsim.hemisphere_polarized surface_normal (%10.4f, %10.4f, %10.4f) \n", surface_normal.x, surface_normal.y, surface_normal.z );  

    float phi = curand_uniform(&rng)*2.f*M_PIf;  // 0->2pi
    float cosTheta = curand_uniform(&rng) ;      // 0->1
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);

    direction->x = cosf(phi)*sinTheta ; 
    direction->y = sinf(phi)*sinTheta ; 
    direction->z = cosTheta ; 

    rotateUz( *direction, (*normal) * ( inwards ? -1.f : 1.f )); 

    // what about normal incidence ?
    const float3 transverse = normalize(cross(*direction, (*normal) * ( inwards ? -1.f : 1.f )  )) ; // perpendicular to plane of incidence
    const float3 within = normalize( cross(*direction, transverse) );  //   within plane of incidence and perpendicular to direction


    switch(polz)
    {
        case 0: *polarization = transverse ; break ;   // S-polarizatiom
        case 1: *polarization = within     ; break ;   // P-polarization
        case 2: *polarization = normalize( 0.5f*transverse + (1.f-0.5f)*within )  ; break ;  // equal admixture
    }
}





template <typename T>
inline QSIM_METHOD float qsim<T>::scint_wavelength_hd0(curandStateXORWOW& rng) 
{
    constexpr float y0 = 0.5f/3.f ; 
    float u0 = curand_uniform(&rng); 
    return tex2D<float>(scint_tex, u0, y0 ); 
}

/**
qsim::scint_wavelength_hd10
--------------------------------------------------

Idea is to improve handling of extremes by throwing ten times the bins
at those regions, using simple and cheap linear mappings.

TODO: move hd "layers" into float4 payload so the 2d cerenkov and 1d scintillation
icdf texture can share some of teh implementation

**/

template <typename T>
inline QSIM_METHOD float qsim<T>::scint_wavelength_hd10(curandStateXORWOW& rng) 
{
    float u0 = curand_uniform(&rng); 
    float wl ; 

    constexpr float y0 = 0.5f/3.f ; 
    constexpr float y1 = 1.5f/3.f ; 
    constexpr float y2 = 2.5f/3.f ; 

    if( u0 < 0.1f )
    {
        wl = tex2D<float>(scint_tex, u0*10.f , y1 );    
    }
    else if ( u0 > 0.9f )
    {
        wl = tex2D<float>(scint_tex, (u0 - 0.9f)*10.f , y2 );    
    }
    else
    {
        wl = tex2D<float>(scint_tex, u0,  y0 ); 
    }
    return wl ; 
}



template <typename T>
inline QSIM_METHOD float qsim<T>::scint_wavelength_hd20(curandStateXORWOW& rng) 
{
    float u0 = curand_uniform(&rng); 
    float wl ; 

    constexpr float y0 = 0.5f/3.f ; 
    constexpr float y1 = 1.5f/3.f ; 
    constexpr float y2 = 2.5f/3.f ; 

    if( u0 < 0.05f )
    {
        wl = tex2D<float>(scint_tex, u0*20.f , y1 );    
    }
    else if ( u0 > 0.95f )
    {
        wl = tex2D<float>(scint_tex, (u0 - 0.95f)*20.f , y2 );    
    }
    else
    {
        wl = tex2D<float>(scint_tex, u0,  y0 ); 
    }
    return wl ; 
}

/**
qsim::cerenkov_wavelength_rejection_sampled
---------------------------------------------

wavelength between Wmin and Wmax is uniform-reciprocal-sampled 
to mimic uniform energy range sampling without taking reciprocals
twice

g4-cls G4Cerenkov::

    251   G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    252   G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    253   G4double dp = Pmax - Pmin;
    254 
    255   G4double nMax = Rindex->GetMaxValue();
    256 
    257   G4double BetaInverse = 1./beta;
    258 
    259   G4double maxCos = BetaInverse / nMax;
    260   G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
    261 
    ...
    270   for (G4int i = 0; i < fNumPhotons; i++) {
    271   
    272       // Determine photon energy
    273   
    274       G4double rand;
    275       G4double sampledEnergy, sampledRI;
    276       G4double cosTheta, sin2Theta;
    277 
    278       // sample an energy
    279 
    280       do {
    281          rand = G4UniformRand();
    282          sampledEnergy = Pmin + rand * dp;                // linear energy sample in Pmin -> Pmax
    283          sampledRI = Rindex->Value(sampledEnergy);
    284          cosTheta = BetaInverse / sampledRI;              // what cone angle that energy sample corresponds to 
    285 
    286          sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);   
    287          rand = G4UniformRand();
    288 
    289         // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
    290       } while (rand*maxSin2 > sin2Theta);                // constrain   
    291 

::

                        
                        
                  \    B    /                        
              \    .   |   .    /                            AC     ct / n          1         i       BetaInverse 
          \    C       |       C    /             cos th =  ---- =  --------   =  ------ =   ---  =  -------------
      \    .    \      |      /    .     /                   AB       bct           b n       n        sampledRI
       .         \    bct    /          .
                  \    |    /                                  BetaInverse
                   \   |   /  ct                  maxCos  =  ----------------- 
                    \  |th/  ----                                nMax                                                
                     \ | /    n
                      \|/
                       A

    Particle travels AB, light travels AC,  ACB is right angle 


     Only get Cerenkov radiation when   

            cos th <= 1 , 

            beta >= beta_min = 1/n        BetaInverse <= BetaInverse_max = n 


     At the small beta threshold AB = AC,   beta = beta_min = 1/n     eg for n = 1.333, beta_min = 0.75  

            cos th = 1,  th = 0         light is in direction of the particle 


     For ultra relativistic particle beta = 1, there is a maximum angle 

            th = arccos( 1/n )      

    In [5]: np.arccos(0.75)*180./np.pi
    Out[5]: 41.40962210927086


     So the beta range to have Cerenkov is  : 

                1/n       slowest, cos th = 1, th = 0   

          ->    1         ultra-relativistic, maximum cone angle th  arccos(1/n)     
 

     Corresponds to BetaInverse range 

           BetaInverse =  n            slowest, cos th = 1, th = 0    cone in particle direction 
    
           BetaInverse  = 1           



                
     The above considers a fixed refractive index.
     Actually refractive index varies with wavelength resulting in 
     a range of cone angles for a fixed particle beta.


    * https://www2.physics.ox.ac.uk/sites/default/files/2013-08-20/external_pergamon_jelley_pdf_18410.pdf
    * ~/opticks_refs/external_pergamon_jelley_pdf_18410.pdf


**/

template <typename T>
inline QSIM_METHOD float qsim<T>::cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng, const GS& g) 
{
    float u0 ;
    float u1 ; 
    float w ; 
    float wavelength ;

    float sampledRI ;
    float cosTheta ;
    float sin2Theta ;
    float u_maxSin2 ;

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)

    do {
        u0 = curand_uniform(&rng) ;

        w = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        wavelength = g.ck1.Wmin*g.ck1.Wmax/w ;  

        float4 props = boundary_lookup(wavelength, line, 0u); 

        sampledRI = props.x ;


        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  // avoid going -ve 

        u1 = curand_uniform(&rng) ;

        u_maxSin2 = u1*g.ck1.maxSin2 ;

    } while ( u_maxSin2 > sin2Theta);


    if( id == 0u )
    {
        printf("// qsim::cerenkov_wavelength_rejection_sampled id %d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f \n", id, sampledRI, cosTheta, sin2Theta, wavelength );  
    }

    return wavelength ; 
}

/**
FOR NOW NOT THE USUAL PHOTON : BUT DEBUGGING THE WAVELENGTH SAMPLING 
**/



template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id )
{
    float u0 ;
    float u1 ; 


    float w_linear ; 
    float wavelength ;

    float sampledRI ;
    float cosTheta ;
    float sin2Theta ;
    float u_mxs2_s2 ;

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)

    unsigned loop = 0u ; 

    do {

#ifdef FLIP_RANDOM
        u0 = 1.f - curand_uniform(&rng) ;
#else
        u0 = curand_uniform(&rng) ;
#endif

        w_linear = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        wavelength = g.ck1.Wmin*g.ck1.Wmax/w_linear ;  

        float4 props = boundary_lookup( wavelength, line, 0u); 

        sampledRI = props.x ;

        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = (1.f - cosTheta)*(1.f + cosTheta);  

#ifdef FLIP_RANDOM
        u1 = 1.f - curand_uniform(&rng) ;
#else
        u1 = curand_uniform(&rng) ;
#endif

        u_mxs2_s2 = u1*g.ck1.maxSin2 - sin2Theta ;

        loop += 1 ; 

        if( id == print_id )
        {
            printf("//qsim::cerenkov_photon id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > 0.f );

    float energy = hc_eVnm/wavelength ; 

    p.q0.f.x = energy ; 
    p.q0.f.y = wavelength ; 
    p.q0.f.z = sampledRI ; 
    p.q0.f.w = cosTheta ; 

    p.q1.f.x = sin2Theta ; 
    p.q1.u.y = 0u ; 
    p.q1.u.z = 0u ; 
    p.q1.f.w = g.ck1.BetaInverse ; 

    p.q2.f.x = w_linear ;    // linear sampled wavelenth
    p.q2.f.y = wavelength ;  // reciprocalized trick : does it really work  
    p.q2.f.z = u0 ; 
    p.q2.f.w = u1 ; 

    p.q3.u.x = line ; 
    p.q3.u.y = loop ; 
    p.q3.f.z = 0.f ; 
    p.q3.f.w = 0.f ; 
} 



/**
qsim::cerenkov_photon_enprop
------------------------------

Variation assuming Wmin, Wmax contain Pmin Pmax and using qprop::interpolate 
to sample the RINDEX

**/




template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, const GS& g, int print_id )
{
    T u0 ;
    T u1 ; 
    T energy ; 
    T sampledRI ;
    T cosTheta ;
    T sin2Theta ;
    T u_mxs2_s2 ;

    // should be MaterialLine no ?
    unsigned line = g.st.MaterialIndex ; //   line :  4*boundary_idx + OMAT/IMAT (0/3)

    unsigned loop = 0u ; 

    do {

        u0 = qcurand<T>::uniform(&rng) ;

        energy = g.ck1.Wmin + u0*(g.ck1.Wmax - g.ck1.Wmin) ; 

        sampledRI = prop->interpolate( 0u, energy ); 

        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = (one - cosTheta)*(one + cosTheta);  

        u1 = qcurand<T>::uniform(&rng) ;

        u_mxs2_s2 = u1*g.ck1.maxSin2 - sin2Theta ;

        loop += 1 ; 

        if( id == print_id )
        {
            printf("//qsim::cerenkov_photon_enprop id %d loop %3d u0 %10.5f ri %10.5f ct %10.5f s2 %10.5f u_mxs2_s2 %10.5f \n", id, loop, u0, sampledRI, cosTheta, sin2Theta, u_mxs2_s2 );
        }


    } while ( u_mxs2_s2 > 0.f );


    float wavelength = hc_eVnm/energy ; 

    p.q0.f.x = energy ; 
    p.q0.f.y = wavelength ; 
    p.q0.f.z = sampledRI ; 
    p.q0.f.w = cosTheta ; 

    p.q1.f.x = sin2Theta ; 
    p.q1.u.y = 0u ; 
    p.q1.u.z = 0u ; 
    p.q1.f.w = g.ck1.BetaInverse ; 

    p.q2.f.x = 0.f ; 
    p.q2.f.y = 0.f ; 
    p.q2.f.z = u0 ; 
    p.q2.f.w = u1 ; 

    p.q3.u.x = line ; 
    p.q3.u.y = loop ; 
    p.q3.f.z = 0.f ; 
    p.q3.f.w = 0.f ; 
} 









/**
qsim::cerenkov_photon_expt
-------------------------------------

This does the sampling all in double, narrowing to 
float just for the photon output.

Note that this is not using a genstep.

Which things have most need to be  double to make any difference ?

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon_expt(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id )
{
    double BetaInverse = 1.5 ; 
    double Pmin = 1.55 ; 
    double Pmax = 15.5 ; 
    double nMax = 1.793 ; 

    //double maxOneMinusCosTheta = (nMax - BetaInverse) / nMax;   

    double maxCos = BetaInverse / nMax;
    double maxSin2 = ( 1. - maxCos )*( 1. + maxCos ); 
    double cosTheta ;
    double sin2Theta ;

    double reject ;
    double u0 ;
    double u1 ; 
    double energy ; 
    double sampledRI ;

    //double oneMinusCosTheta ;

    unsigned loop = 0u ; 

    do {
        u0 = curand_uniform_double(&rng) ;
        u1 = curand_uniform_double(&rng) ;
        energy = Pmin + u0*(Pmax - Pmin) ; 
        sampledRI = prop->interpolate( 0u, energy ); 
        //oneMinusCosTheta = (sampledRI - BetaInverse) / sampledRI ; 
        //reject = u1*maxOneMinusCosTheta - oneMinusCosTheta ;
        loop += 1 ; 

        cosTheta = BetaInverse / sampledRI ;
        sin2Theta = (1. - cosTheta)*(1. + cosTheta);  
        reject = u1*maxSin2 - sin2Theta ;

    } while ( reject > 0. );


    // narrowing for output 
    p.q0.f.x = energy ; 
    p.q0.f.y = hc_eVnm/energy ;
    p.q0.f.z = sampledRI ; 
    //p.q0.f.w = 1. - oneMinusCosTheta ; 
    p.q0.f.w = cosTheta ; 

    p.q1.f.x = sin2Theta ; 
    p.q1.u.y = 0u ; 
    p.q1.u.z = 0u ; 
    p.q1.f.w = BetaInverse ; 

    p.q2.f.x = reject ; 
    p.q2.f.y = 0.f ; 
    p.q2.f.z = u0 ; 
    p.q2.f.w = u1 ; 

    p.q3.f.x = 0.f ; 
    p.q3.u.y = loop ; 
    p.q3.f.z = 0.f ; 
    p.q3.f.w = 0.f ; 
} 



/**
qsim::generate_photon_carrier
------------------------------

An input photon carried within the genstep q2:q5 is repeatedly provided. 

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_carrier(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
{
    p.q0.f = gs.q2.f ; 
    p.q1.f = gs.q3.f ; 
    p.q2.f = gs.q4.f ; 
    p.q3.f = gs.q5.f ; 

    p.set_flag(TORCH); 
}

/**
qsim::generate_photon_simtrace
--------------------------------

* NB simtrace cxs center-extent-genstep are very different to standard Cerenkov/Scintillation gensteps 

These gensteps are for example created in SEvent::MakeCenterExtentGensteps the below generation 
should be comparable to the CPU implementation SEvent::GenerateCenterExtentGenstepsPhotons

HMM: note that the photon_id is global to the launch making it potentially a very large number 
but for identication purposes having a local to the photons of the genstep index would
be more useful and would allow storage within much less bits.

TODO: implement local index by including photon_id offset with the gensteps 

* NB the sevent.h enum order is different to the python one  eg XYZ=0 
**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_simtrace(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    C4U gsid ;  

    //int gencode          = gs.q0.i.x ;   
    int gridaxes           = gs.q0.i.y ;  // { XYZ, YZ, XZ, XY }
    gsid.u                 = gs.q0.i.z ; 
    //unsigned num_photons = gs.q0.u.w ; 


    p.q0.f.x = gs.q1.f.x ;   // start with genstep local frame position, typically origin  (0,0,0)   
    p.q0.f.y = gs.q1.f.y ; 
    p.q0.f.z = gs.q1.f.z ; 
    p.q0.f.w = 1.f ;        

    //printf("//qsim.generate_photon_torch gridaxes %d gs.q1 (%10.4f %10.4f %10.4f %10.4f) \n", gridaxes, gs.q1.f.x, gs.q1.f.y, gs.q1.f.z, gs.q1.f.w ); 

    float u0 = curand_uniform(&rng); 

    float sinPhi, cosPhi;
    sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);

    float u1 = curand_uniform(&rng); 
    float cosTheta = 2.f*u1 - 1.f ; 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta) ; 

    //printf("//qsim.generate_photon_torch u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f \n", u0, sinPhi, cosPhi ); 
    //printf("//qsim.generate_photon_torch u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n", u1, sinTheta, cosTheta ); 
    //printf("//qsim.generate_photon_torch  u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n",  u0, sinPhi, cosPhi, u1, sinTheta, cosTheta ); 

    switch( gridaxes )
    { 
        case YZ:  { p.q1.f.x = 0.f    ;  p.q1.f.y = cosPhi ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ; 
        case XZ:  { p.q1.f.x = cosPhi ;  p.q1.f.y = 0.f    ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ; 
        case XY:  { p.q1.f.x = cosPhi ;  p.q1.f.y = sinPhi ;  p.q1.f.z = 0.f    ;  p.q1.f.w = 0.f ; } ; break ; 
        case XYZ: { p.q1.f.x = sinTheta*cosPhi ;  
                    p.q1.f.y = sinTheta*sinPhi ;  
                    p.q1.f.z = cosTheta        ;  
                    p.q1.f.w = 0.f ; } ; break ;   // previously used XZ
    }


    qat4 qt(gs) ; // copy 4x4 transform from last 4 quads of genstep 
    qt.right_multiply_inplace( p.q0.f, 1.f );   // position 
    qt.right_multiply_inplace( p.q1.f, 0.f );   // direction 


    unsigned char ucj = (photon_id < 255 ? photon_id : 255 ) ;
    gsid.c4.w = ucj ; 
    p.q3.u.w = gsid.u ;

} 



/**
qsim::generate_photon
----------------------

Moved non-standard center-extent gensteps to use qsim::generate_photon_simtrace not this 

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    const int& gencode = gs.q0.i.x ; 
    //printf("//qsim.generate_photon gencode %d \n", gencode); 
    switch(gencode)
    {
        case OpticksGenstep_PHOTON_CARRIER:  generate_photon_carrier(p, rng, gs, photon_id, genstep_id) ; break ; 
        default:                             generate_photon_dummy(p, rng, gs, photon_id, genstep_id) ; break ; 
    }
}





/**
qsim::cerenkov_wavelength with a fabricated genstep for testing
-----------------------------------------------------------------

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_fabricate_genstep(GS& g, bool energy_range )
{
    // picks the material line from which to get RINDEX
    unsigned MaterialLine = boundary_tex_MaterialLine_LS ;  
    float nMax = 1.793f ; 
    float BetaInverse = 1.500f ; 

    float maxCos = BetaInverse / nMax;
    float maxSin2 = (1.f - maxCos) * (1.f + maxCos) ;

    g.st.Id = 0 ; 
    g.st.ParentId = 0 ; 
    g.st.MaterialIndex = MaterialLine ; 
    g.st.NumPhotons = 0 ; 

    g.st.x0.x = 100.f ; 
    g.st.x0.y = 100.f ; 
    g.st.x0.z = 100.f ; 
    g.st.t0 = 20.f ; 

    g.st.DeltaPosition.x = 1000.f ; 
    g.st.DeltaPosition.y = 1000.f ; 
    g.st.DeltaPosition.z = 1000.f ; 
    g.st.step_length = 1000.f ; 

    g.ck1.code = 0 ; 
    g.ck1.charge = 1.f ; 
    g.ck1.weight = 1.f ; 
    g.ck1.preVelocity = 0.f ; 

    float Pmin = 1.55f ; 
    float Pmax = 15.5f ; 

    g.ck1.BetaInverse = BetaInverse ;      //  g.ck1.BetaInverse/sampledRI  : yields the cone angle cosTheta

    if(energy_range)
    {
        g.ck1.Wmin = Pmin ;    
        g.ck1.Wmax = Pmax ; 
    }
    else
    {
        g.ck1.Wmin = hc_eVnm/Pmax ;            // close to: 1240./15.5 = 80.               
        g.ck1.Wmax = hc_eVnm/Pmin ;            // close to: 1240./1.55 = 800.              
    }

    g.ck1.maxCos = maxCos  ;               //  is this used?          

    g.ck1.maxSin2 = maxSin2 ;              // constrains cone angle rejection sampling   
    g.ck1.MeanNumberOfPhotons1 = 0.f ; 
    g.ck1.MeanNumberOfPhotons2 = 0.f ; 
    g.ck1.postVelocity = 0.f ; 

} 


template <typename T>
inline QSIM_METHOD float qsim<T>::cerenkov_wavelength_rejection_sampled(unsigned id, curandStateXORWOW& rng ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(g, energy_range); 
    float wavelength = cerenkov_wavelength_rejection_sampled(id, rng, g);   
    return wavelength ; 
}

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = false ; 
    cerenkov_fabricate_genstep(g, energy_range); 
    cerenkov_photon(p, id, rng, g, print_id); 
}

template <typename T>
inline QSIM_METHOD void qsim<T>::cerenkov_photon_enprop(quad4& p, unsigned id, curandStateXORWOW& rng, int print_id ) 
{
    QG qg ;      
    qg.zero();  
    GS& g = qg.g ; 
    bool energy_range = true ; 
    cerenkov_fabricate_genstep(g, energy_range); 
    cerenkov_photon_enprop(p, id, rng, g, print_id); 
}







/**
qsim::scint_dirpol
--------------------

Fills the photon quad4 struct with the below:

* direction, weight
* polarization, wavelength 

NB no position, time.

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::scint_dirpol(quad4& p, curandStateXORWOW& rng)
{
    float u0 = curand_uniform(&rng) ; 
    float u1 = curand_uniform(&rng) ; 
    float u2 = curand_uniform(&rng) ;   
    float u3 = curand_uniform(&rng) ;   

    float ct = 1.0f - 2.0f*u1 ;                 // -1.: 1. 
    float st = sqrtf( (1.0f-ct)*(1.0f+ct)) ; 
    float phi = 2.f*M_PIf*u2 ;
    float sp = sinf(phi); 
    float cp = cosf(phi); 
    float3 dir0 = make_float3( st*cp, st*sp,  ct ); 

    p.q1.f.x = dir0.x ; 
    p.q1.f.y = dir0.y ; 
    p.q1.f.z = dir0.z ; 
    p.q1.f.w = 1.f ;    // weight   

    float3 pol0 = make_float3( ct*cp, ct*sp, -st );
    float3 perp = cross( dir0, pol0 ); 
    float az =  2.f*M_PIf*u3 ; 
    float sz = sin(az);
    float cz = cos(az);
    float3 pol1 = normalize( cz*pol0 + sz*perp ) ; 

    p.q2.f.x = pol1.x ; 
    p.q2.f.y = pol1.y ; 
    p.q2.f.z = pol1.z ; 
    p.q2.f.w = scint_wavelength_hd20(rng); // hmm should this switch on hd_factor  
}

/**
Because reemission is possible (inside scintillators) for photons arising from Cerenkov (or Torch) 
gensteps need to special case handle the scintillationTime somehow ? 

Could carry the single float (could be domain compressed, it is eg 1.5 ns) in other gensteps ? 
But it is material specific just like REEMISSIONPROB so its more appropriate 
to live in the boundary_tex alongside the REEMISSIONPROB. 
But it could be carried in the genstep(or anywhere) as its use is "gated" by a non-zero REEMISSIONPROB.

Prefer to just hold it in the context, and provide G4Opticks::setReemissionScintillationTime API 
for setting it (default 0.) that is used from detector specific code which can read from 
the Geant4 properties directly.  What about geocache ? Can hold/persist with GScintillatorLib metadata.

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng)
{
    scint_dirpol(p, rng); 
    float u4 = curand_uniform(&rng) ; 
    p.q0.f.w += -scintillationTime*logf(u4) ;
}

template <typename T>
inline QSIM_METHOD void qsim<T>::scint_photon(quad4& p, GS& g, curandStateXORWOW& rng)
{
    p.zero(); 
    scint_dirpol(p, rng); 

    float fraction = g.sc1.charge == 0.f  ? 1.f : curand_uniform(&rng) ;   
    float u4 = curand_uniform(&rng) ; 

    p.q0.f.x = g.st.x0.x + fraction*g.st.DeltaPosition.x ; 
    p.q0.f.y = g.st.x0.y + fraction*g.st.DeltaPosition.y ; 
    p.q0.f.z = g.st.x0.z + fraction*g.st.DeltaPosition.z ; 
    p.q0.f.w = g.st.t0   + fraction*g.st.step_length/g.sc1.midVelocity - g.sc1.ScintillationTime*logf(u4) ;
}


template <typename T>
inline QSIM_METHOD void qsim<T>::scint_photon(quad4& p, curandStateXORWOW& rng)
{
    QG qg ;      
    qg.zero();  

    GS& g = qg.g ; 

    // fabricate some values for the genstep
    g.st.Id = 0 ; 
    g.st.ParentId = 0 ; 
    g.st.MaterialIndex = 0 ; 
    g.st.NumPhotons = 0 ; 

    g.st.x0.x = 100.f ; 
    g.st.x0.y = 100.f ; 
    g.st.x0.z = 100.f ; 
    g.st.t0 = 20.f ; 

    g.st.DeltaPosition.x = 1000.f ; 
    g.st.DeltaPosition.y = 1000.f ; 
    g.st.DeltaPosition.z = 1000.f ; 
    g.st.step_length = 1000.f ; 

    g.sc1.code = 1 ; 
    g.sc1.charge = 1.f ;
    g.sc1.weight = 1.f ;
    g.sc1.midVelocity = 0.f ; 

    g.sc1.scnt = 0 ;
    g.sc1.f41 = 0.f ;   
    g.sc1.f42 = 0.f ;   
    g.sc1.f43 = 0.f ;   

    g.sc1.ScintillationTime = 10.f ;
    g.sc1.f51 = 0.f ;
    g.sc1.f52 = 0.f ;
    g.sc1.f53 = 0.f ;

    scint_photon(p, g, rng); 
}





#endif

