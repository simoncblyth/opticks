#pragma once
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

TODO:

0. many of the below methods could be static (or const)
1. some methods have too many parameters that could be avoided using 
   some carefully chosen members, eg print_id/pidx
2. get more methods to work on CPU as well as GPU for easier testing 
   NB must move decl and implementation to do this 
3. more encapsulation of sub-concerns eg boundary 

**/


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
#include "sevent.h"  
#include "sphoton.h"

#include "storch.h"
#include "scarrier.h"

#include "srec.h"
#include "sseq.h"
#include "scurand.h"

#include "qbase.h"
#include "qevent.h"
#include "qprop.h"
#include "qmultifilm.h"
#include "qbnd.h"
#include "qstate.h"

#include "qscint.h"
#include "qcerenkov.h"

struct curandStateXORWOW ; 
struct qcerenkov ; 

struct qsim
{
    qbase*              base ; 
    qevent*             evt ; 
    curandStateXORWOW*  rngstate ; 
    qbnd*               bnd ; 
    qmultifilm*         multifilm;
    qcerenkov*          cerenkov ; 
    qscint*             scint ; 
            

    QSIM_METHOD void    generate_photon_dummy( quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD static float3 uniform_sphere(const float u0, const float u1); 


#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND )
    QSIM_METHOD static float3 uniform_sphere(curandStateXORWOW& rng); 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)

    QSIM_METHOD float4  multifilm_lookup(unsigned pmtType, unsigned boundary, float nm, float aoi);

    QSIM_METHOD void    generate_photon_carrier(    quad4&   p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD void    generate_photon_simtrace(   quad4&   p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD void    generate_photon(            sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 


    QSIM_METHOD static void lambertian_direction(float3* dir, const float3* normal, float orient, curandStateXORWOW& rng, unsigned idx  ); 
    QSIM_METHOD static void random_direction_marsaglia(float3* dir, curandStateXORWOW& rng, unsigned idx); 

    QSIM_METHOD static void rayleigh_scatter(sphoton& p, curandStateXORWOW& rng ); 


    QSIM_METHOD void    mock_propagate( sphoton& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx ); 

    QSIM_METHOD int     propagate(const int bounce, sphoton& p, qstate& s, const quad2* prd, curandStateXORWOW& rng, unsigned idx ); 
    QSIM_METHOD int     propagate_to_boundary(unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx); 
    QSIM_METHOD int     propagate_at_surface( unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx); 
    QSIM_METHOD int     propagate_at_boundary(unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx); 

    QSIM_METHOD void    reflect_diffuse(  sphoton& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx );
    QSIM_METHOD void    reflect_specular( sphoton& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx );

    QSIM_METHOD void    hemisphere_polarized( sphoton& p, unsigned polz, bool inwards, const quad2* prd, curandStateXORWOW& rng); 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    qsim()    // instanciated on CPU (see QSim::init_sim) and copied to device so no ctor in device code
        :
        base(nullptr),
        evt(nullptr),
        rngstate(nullptr),
        bnd(nullptr),
        multifilm(nullptr),
        cerenkov(nullptr),
        scint(nullptr)
    {
    }
#endif

}; 


inline QSIM_METHOD void qsim::generate_photon_dummy(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    //printf("//qsim::generate_photon_dummy photon_id %d ", photon_id ); 
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



#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND )

inline QSIM_METHOD float3 qsim::uniform_sphere(curandStateXORWOW& rng)
{
    float phi = curand_uniform(&rng)*2.f*M_PIf;
    float cosTheta = 2.f*curand_uniform(&rng) - 1.f ; // -1.f -> 1.f 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);
    return make_float3(cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta); 
}

#endif


inline QSIM_METHOD float3 qsim::uniform_sphere(const float u0, const float u1)
{
    float phi = u0*2.f*M_PIf;
    float cosTheta = 2.f*u1 - 1.f ; // -1.f -> 1.f 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);
    return make_float3(cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta); 
}



// TODO: get more of the below to work on CPU with mocked curand (and in future mocked tex2D and cudaTextureObject_t )

#if defined(__CUDACC__) || defined(__CUDABE__)


/*
 qsim::multifilm_lookup
-------------------
 
*/

inline QSIM_METHOD float4 qsim::multifilm_lookup(unsigned pmtType, unsigned boundary, float nm, float aoi){

    float4 value = multifilm->lookup(pmtType, boundary, nm, aoi);
    return value;
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
inline  QSIM_METHOD void qsim::lambertian_direction(float3* dir, const float3* normal, float orient, curandStateXORWOW& rng, unsigned idx )
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


inline QSIM_METHOD void qsim::random_direction_marsaglia(float3* dir,  curandStateXORWOW& rng, unsigned idx )
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
qsim::rayleigh_scatter
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

inline QSIM_METHOD void qsim::rayleigh_scatter(sphoton& p, curandStateXORWOW& rng )
{
    float3* p_direction = &p.mom ; 
    float3* p_polarization = &p.pol ;

 
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

        smath::rotateUz(direction, *p_direction );

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

            smath::rotateUz(polarization, direction);
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

inline QSIM_METHOD int qsim::propagate_to_boundary(unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
{
    const float& absorption_length = s.material1.y ; 
    const float& scattering_length = s.material1.z ; 
    const float& reemission_prob = s.material1.w ; 
    const float& group_velocity = s.m1group2.x ; 
    const float& distance_to_boundary = prd->q0.f.w ; 

    float u_scattering = curand_uniform(&rng) ;
    float u_absorption = curand_uniform(&rng) ;
    float scattering_distance = -scattering_length*logf(u_scattering);   
    float absorption_distance = -absorption_length*logf(u_absorption);

#ifdef DEBUG_TIME
    if( idx == pidx ) printf("//qsim.propagate_to_boundary[ idx %d post (%10.4f %10.4f %10.4f %10.4f) \n", idx, p.pos.x, p.pos.y, p.pos.z, p.time );  
#endif

#ifdef DEBUG_HIST
    if(idx == pidx ) printf("//qsim.propagate_to_boundary idx %d distance_to_boundary %10.4f absorption_distance %10.4f scattering_distance %10.4f u_scattering %10.4f u_absorption %10.4f \n", 
             idx, distance_to_boundary, absorption_distance, scattering_distance, u_scattering, u_absorption  ); 
#endif
  

    if (absorption_distance <= scattering_distance) 
    {   
        if (absorption_distance <= distance_to_boundary) 
        {   
            p.time += absorption_distance/group_velocity ;   
            p.pos  += absorption_distance*(p.mom) ;

#ifdef DEBUG_TIME
            float absorb_time_delta = absorption_distance/group_velocity ; 
            if( idx == pidx ) printf("//qsim.propagate_to_boundary] idx %d post (%10.4f %10.4f %10.4f %10.4f) absorb_time_delta %10.4f   \n", 
                         idx, p.pos.x, p.pos.y, p.pos.z, p.time, absorb_time_delta  );  
#endif

            float u_reemit = reemission_prob == 0.f ? 2.f : curand_uniform(&rng);  // avoid consumption at absorption when not scintillator

            if (u_reemit < reemission_prob)    
            {   
                p.wavelength = scint->wavelength(rng);
                p.mom = uniform_sphere(rng);
                p.pol = normalize(cross(uniform_sphere(rng), p.mom));

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
            p.time += scattering_distance/group_velocity ;
            p.pos  += scattering_distance*(p.mom) ;

            rayleigh_scatter(p, rng); // changes dir and pol, consumes 5u at each turn of rejection sampling loop

            flag = BULK_SCATTER;

            return CONTINUE;
        }       
          //  otherwise sail to boundary  
    }     // if scattering_distance < absorption_distance



    p.pos  += distance_to_boundary*(p.mom) ;
    p.time += distance_to_boundary/group_velocity   ;  

#ifdef DEBUG_TIME
    float sail_time_delta = distance_to_boundary/group_velocity ; 
    if( idx == pidx ) printf("//qsim.propagate_to_boundary] idx %d post (%10.4f %10.4f %10.4f %10.4f) sail_time_delta %10.4f   \n", 
          idx, p.pos.x, p.pos.y, p.pos.z, p.time, sail_time_delta  );  
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

inline QSIM_METHOD int qsim::propagate_at_boundary(unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
{
    const float& n1 = s.material1.x ;
    const float& n2 = s.material2.x ;   
    const float eta = n1/n2 ; 

    const float3* normal = (float3*)&prd->q0.f.x ; 

    const float _c1 = -dot(p.mom, *normal ); 
    const float3 oriented_normal = _c1 < 0.f ? -(*normal) : (*normal) ; 
    const float c1 = fabs(_c1) ; 
    const bool normal_incidence = c1 == 1.f ; 

    /* 
    printf("//qsim.propagate_at_boundary idx %d nrm   (%10.4f %10.4f %10.4f) \n", idx, oriented_normal.x, oriented_normal.y, oriented_normal.z ); 
    printf("//qsim.propagate_at_boundary idx %d mom_0 (%10.4f %10.4f %10.4f) \n", idx, p.mom.x, p.mom.y, p.mom.z ); 
    printf("//qsim.propagate_at_boundary idx %d pol_0 (%10.4f %10.4f %10.4f) \n", idx, p.pol.x, p.pol.y, p.pol.z ); 
    printf("//qsim.propagate_at_boundary idx %d c1 %10.4f normal_incidence %d \n", idx, c1, normal_incidence ); 
    */

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law and trig identity 
    bool tir = c2c2 < 0.f ; 
    const float EdotN = dot(p.pol, oriented_normal ) ;  // used for TIR polarization
    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect
    const float n1c1 = n1*c1 ;
    const float n2c2 = n2*c2 ; 
    const float n2c1 = n2*c1 ; 
    const float n1c2 = n1*c2 ; 
    const float3 A_trans = normal_incidence ? p.pol : normalize(cross(p.mom, oriented_normal)) ; // perpendicular to plane of incidence
    const float E1_perp = dot(p.pol, A_trans);     //  E vector component perpendicular to plane of incidence, ie S polarization
    const float2 E1   = normal_incidence ? make_float2( 0.f, 1.f) : make_float2( E1_perp , length( p.pol - (E1_perp*A_trans) ) ); 
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


    p.mom = reflect
                    ?
                       p.mom + 2.0f*c1*oriented_normal
                    :
                       eta*(p.mom) + (eta*c1 - c2)*oriented_normal
                    ;


    const float3 A_paral = normalize(cross(p.mom, A_trans));

    p.pol =  normal_incidence ?
                                         ( reflect ?  p.pol*(n2>n1? -1.f:1.f) : p.pol )
                                      : 
                                         ( reflect ?
                                                   ( tir ?  -p.pol + 2.f*EdotN*oriented_normal : RR.x*A_trans + RR.y*A_paral )

                                                   :
                                                       TT.x*A_trans + TT.y*A_paral 
                                             
                                                   )
                                      ;

    /*
    printf("//qsim.propagate_at_boundary idx %d reflect %d tir %d TransCoeff %10.4f u_reflect %10.4f \n", idx, reflect, tir, TransCoeff, u_reflect );  
    printf("//qsim.propagate_at_boundary idx %d mom_1 (%10.4f %10.4f %10.4f) \n", idx, p.mom.x, p.mom.y, p.mom.z ); 
    printf("//qsim.propagate_at_boundary idx %d pol_1 (%10.4f %10.4f %10.4f) \n", idx, p.pol.x, p.pol.y, p.pol.z ); 
    */

    /*
    if(idx == 251959)
    {
        printf("//qsim.propagate_at_boundary RR.x %10.4f A_trans (%10.4f %10.4f %10.4f )  RR.y %10.4f  A_paral (%10.4f %10.4f %10.4f ) \n", 
              RR.x, A_trans.x, A_trans.y, A_trans.z,
              RR.y, A_paral.x, A_paral.y, A_paral.z ); 

        printf("//qsim.propagate_at_boundary reflect %d  tir %d polarization (%10.4f, %10.4f, %10.4f) \n", reflect, tir, p.pol.x, p.pol.y, p.pol.z );  
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

inline QSIM_METHOD int qsim::propagate_at_surface(unsigned& flag, sphoton& p, const quad2* prd, const qstate& s, curandStateXORWOW& rng, unsigned idx)
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

inline QSIM_METHOD void qsim::reflect_diffuse( sphoton& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx )
{
    float3 old_mom = p.mom ; 

    const float3* normal = prd->normal()  ;  
    const float orient = -1.f ;     // equivalent to G4OpBoundaryProcess::PostStepDoIt early flip  of theGlobalNormal ?
    lambertian_direction( &p.mom, normal, orient, rng, idx );

    float3 facet_normal = normalize( p.mom - old_mom ); 
    const float EdotN = dot( p.pol, facet_normal ); 
    p.pol = -1.f*(p.pol) + 2.f*EdotN*facet_normal ; 
}

inline QSIM_METHOD void qsim::reflect_specular( sphoton& p, const quad2* prd, curandStateXORWOW& rng, unsigned idx )
{
    const float3* normal = prd->normal() ;      
    const float orient = -1.f ;     // equivalent to G4OpBoundaryProcess::PostStepDoIt early flip of theGlobalNormal ?

    const float PdotN = dot( p.mom, *normal )*orient ; 
    p.mom = p.mom - 2.f*PdotN*(*normal)*orient ; 

    const float EdotN = dot( p.pol, *normal )*orient ; 
    p.pol = -1.f*(p.pol) + 2.f*EdotN*(*normal)*orient  ; 
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


HMM: seqhis/seqmat should not depend on rec collection, and it must be optional 

**/

inline QSIM_METHOD void qsim::mock_propagate( sphoton& p, const quad2* mock_prd, curandStateXORWOW& rng, unsigned idx )
{
    p.set_flag(TORCH);  // setting initial flag : in reality this should be done by generation


    printf("//qsim.mock_propagate evt.max_bounce %d evt.max_record %d evt.record %p evt.num_record %d evt.num_rec %d \n", 
                                  evt->max_bounce, evt->max_record, evt->record, evt->num_record, evt->num_rec ); 

    int bounce = 0 ; 
    int command = START ; 

    qstate state = {} ; 
    srec rec = {} ;    // compressed step record 
    sseq seq = {} ;  // seqhis..

    // should this state live in evt ? NO, this state must be "thread" local, 
    // the evt instance is shared by all threads and always saves into (idx, bounce) 
    // slotted locations   
    //
    // want the ability to easily eliminate parts of the state that are not needed 
    // via macros : so this current structure manages that simply  

    while( bounce < evt->max_bounce )
    {
         // HMM: encapsulate this step saving 
        //  evt->step(idx, bounce,  

        if(evt->record) evt->record[evt->max_record*idx+bounce] = p ;  
        if(evt->rec)    evt->add_rec(rec, idx, bounce, p ); 
        if(evt->seq)    seq.add_step( bounce, p.flag(), p.boundary() ); 


        const quad2* prd = mock_prd + (evt->max_bounce*idx+bounce) ;  

        command = propagate(bounce, p, state, prd, rng, idx ); 
        bounce++;        

        if(command == BREAK) break ;    
    }

    if(evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;  
    if(evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p ); 
    if(evt->seq    && bounce < evt->max_seq    ) seq.add_step(bounce, p.flag(), p.boundary() ); 

    if(evt->seq) evt->seq[idx] = seq ; 
}

/**
qsim::propagate : one "bounce" propagate_to_boundary/propagate_at_boundary 
-----------------------------------------------------------------------------

This is canonically invoked from within the bounce loop of CSGOptiX/OptiX7Test.cu:simulate 

TODO: missing needs to return BREAK   

**/

inline QSIM_METHOD int qsim::propagate(const int bounce, sphoton& p, qstate& s, const quad2* prd, curandStateXORWOW& rng, unsigned idx ) 
{
    const unsigned boundary = prd->boundary() ; 
    const unsigned identity = prd->identity() ; 
    const float3* normal = prd->normal(); 
    float cosTheta = dot(p.mom, *normal ) ;    

#ifdef DEBUG_COSTHETA
    if( idx == pidx ) printf("//qsim.propagate idx %d bnc %d cosTheta %10.4f dir (%10.4f %10.4f %10.4f) nrm (%10.4f %10.4f %10.4f) \n", 
                 idx, bounce, cosTheta, p.mom.x, p.mom.y, p.mom.z, normal->x, normal->y, normal->z ); 
#endif

    p.set_prd(boundary, identity, cosTheta); 

    bnd->fill_state(s, boundary, p.wavelength, cosTheta, idx ); 

    unsigned flag = 0 ;  

    int command = propagate_to_boundary( flag, p, prd, s, rng, idx ); 
    //if( idx == 0 ) 
    //printf("//qsim.propagate idx %d bounce %d command %d flag %d s.optical.x %d \n", idx, bounce, command, flag, s.optical.x  );   

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

inline QSIM_METHOD void qsim::hemisphere_polarized(sphoton& p, unsigned polz, bool inwards, const quad2* prd, curandStateXORWOW& rng)
{
    const float3* normal = prd->normal() ; 

    //printf("//qsim.hemisphere_polarized polz %d normal (%10.4f, %10.4f, %10.4f) \n", polz, normal->x, normal->y, normal->z );  

    float phi = curand_uniform(&rng)*2.f*M_PIf;  // 0->2pi
    float cosTheta = curand_uniform(&rng) ;      // 0->1
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);

    p.mom.x = cosf(phi)*sinTheta ; 
    p.mom.y = sinf(phi)*sinTheta ; 
    p.mom.z = cosTheta ; 

    smath::rotateUz( p.mom, (*normal) * ( inwards ? -1.f : 1.f )); 

    //printf("//qsim.hemisphere_polarized polz %d p.mom (%10.4f, %10.4f, %10.4f) \n", polz, p.mom.x, p.mom.y, p.mom.z );  

    // what about normal incidence ?
    const float3 transverse = normalize(cross(p.mom, (*normal) * ( inwards ? -1.f : 1.f )  )) ; // perpendicular to plane of incidence
    const float3 within = normalize( cross(p.mom, transverse) );  //   within plane of incidence and perpendicular to direction

    switch(polz)
    {
        case 0: p.pol = transverse ; break ;   // S-polarizatiom
        case 1: p.pol = within     ; break ;   // P-polarization
        case 2: p.pol = normalize( 0.5f*transverse + (1.f-0.5f)*within )  ; break ;  // equal admixture
    }
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

inline QSIM_METHOD void qsim::generate_photon_simtrace(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
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

    //printf("//qsim.generate_photon_simtrace gridaxes %d gs.q1 (%10.4f %10.4f %10.4f %10.4f) \n", gridaxes, gs.q1.f.x, gs.q1.f.y, gs.q1.f.z, gs.q1.f.w ); 

    float u0 = curand_uniform(&rng); 

    float sinPhi, cosPhi;
    sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);

    float u1 = curand_uniform(&rng); 
    float cosTheta = 2.f*u1 - 1.f ; 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta) ; 

    //printf("//qsim.generate_photon_simtrace u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f \n", u0, sinPhi, cosPhi ); 
    //printf("//qsim.generate_photon_simtrace u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n", u1, sinTheta, cosTheta ); 
    //printf("//qsim.generate_photon_simtrace  u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n",  u0, sinPhi, cosPhi, u1, sinTheta, cosTheta ); 

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

inline QSIM_METHOD void qsim::generate_photon(sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    quad4& q = (quad4&)p ; 
    const int& gencode = gs.q0.i.x ; 

    switch(gencode)
    {
        case OpticksGenstep_CARRIER:         scarrier::generate(     q, rng, gs, photon_id, genstep_id)  ; break ; 
        case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ; 
        case OpticksGenstep_CERENKOV:        cerenkov->generate(     p, rng, gs, photon_id, genstep_id ) ; break ; 
        case OpticksGenstep_SCINTILLATION:   scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ; 
        default:                             generate_photon_dummy(  q, rng, gs, photon_id, genstep_id)  ; break ; 
    }
}


#endif

