#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSIM_METHOD __device__
#else
   #define QSIM_METHOD 
#endif 

#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "sqat4.h"
#include "sc4u.h"
#include "sevent.h"

#include "qgs.h"
#include "qprop.h"
#include "qcurand.h"
#include "qbnd.h"
#include "qstate.h"
#include "qprd.h"

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

enum { BREAK, CONTINUE, PASS, START, RETURN }; // return value from propagate_to_boundary

template <typename T>
struct qsim
{
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

    static constexpr float hc_eVnm = 1239.8418754200f ; // G4: h_Planck*c_light/(eV*nm) 
 

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

    QSIM_METHOD void    generate_photon(      quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id  ); 
    QSIM_METHOD void    generate_photon_dummy(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id  ); 
    QSIM_METHOD void    generate_photon_torch(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id  ); 

    QSIM_METHOD void    fill_state(qstate& s, unsigned boundary, float wavelength, float cosTheta ); 

    QSIM_METHOD static float3 uniform_sphere(curandStateXORWOW& rng); 
    QSIM_METHOD static float3 uniform_sphere(const float u0, const float u1); 

    QSIM_METHOD static void   rotateUz(float3& d, const float3& u ); 
    QSIM_METHOD static void   rayleigh_scatter_align(quad4& p, curandStateXORWOW& rng ); 

    QSIM_METHOD int     propagate_to_boundary(unsigned& flag, quad4& p, const qprd& prd, const qstate& s, curandStateXORWOW& rng); 
    QSIM_METHOD int     propagate_at_boundary(                quad4& p, const qprd& prd, const qstate& s, curandStateXORWOW& rng); 


#else
    qsim()
        :
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


// TODO: get the below to work on CPU with mocked curand and tex2D


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

HMM: perhaps simpler not to bother with signing the boundary, just simply use the
cosTheta to give that info at raygen level 

pick relevant boundary_tex lines depening on boundary sign, ie photon direction relative to normal


cosTheta < 0.f 
   photons direction is against the surface normal, ie are entering the shape
   so boundary sign convention is -ve making line+OSUR the relevant surface
   and line+OMAT the relevant first material

cosTheta > 0.f 
   photons direction is with the surface normal, ie are exiting the shape
   so boundary sign convention is +ve making line+ISUR the relevant surface
   and line+IMAT the relevant first material

boundary 
   1 based code, signed by cos_theta of photon direction to outward geometric normal
   >0 outward going photon
   <0 inward going photon


NB the line is above the details of the payload (ie how many float4 per matsur) it is just::
 
    boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 

**/

template <typename T>
inline QSIM_METHOD void qsim<T>::fill_state(qstate& s, unsigned boundary, float wavelength, float cosTheta  )
{
    // HMM: now that are not signing the boundary could use 0-based 

    const int line = (boundary-1)*_BOUNDARY_NUM_MATSUR ;   
    const int m1_line = cosTheta > 0.f ? line + IMAT : line + OMAT ;   
    const int m2_line = cosTheta > 0.f ? line + OMAT : line + IMAT ;   
    const int su_line = cosTheta > 0.f ? line + ISUR : line + OSUR ;   

    //printf("//qsim.fill_state boundary %d line %d wavelength %10.4f m1_line %d \n", boundary, line, wavelength, m1_line ); 

    s.material1 = boundary_lookup( wavelength, m1_line, 0);  
    s.m1group2  = boundary_lookup( wavelength, m1_line, 1);  
    s.material2 = boundary_lookup( wavelength, m2_line, 0); 
    s.surface   = boundary_lookup( wavelength, su_line, 0);    

    // HUH: this would imply the optical buffer is 4 times the length of the bnd ? 
    //     YES it should be and now is see  GBndLib::createOpticalBuffer GBndLin::getOpticalBuf

    s.optical = optical[su_line].u ;   // index/type/finish/value

    s.index.x = optical[m1_line].u.x ; // m1 index
    s.index.y = optical[m2_line].u.x ; // m2 index 
    s.index.z = optical[su_line].u.x ; // su index
    s.index.w = 0u ;                   // avoid undefined memory comparison issues

    //printf("//qsim.fill_state \n"); 
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
uses *u* as its third column and is given by: 

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

could return the flag rather than the action and switch on the flag to continue/break/sail 

**/

template <typename T>
inline QSIM_METHOD int qsim<T>::propagate_to_boundary(unsigned& flag, quad4& p, const qprd& prd, const qstate& s, curandStateXORWOW& rng)
{
    const float& absorption_length = s.material1.y ; 
    const float& scattering_length = s.material1.z ; 
    const float& reemission_prob = s.material1.w ; 
    const float& group_velocity = s.m1group2.x ; 

    const float& distance_to_boundary = prd.t ; 

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

    printf("//qsim.propagate_to_boundary distance_to_boundary %10.4f absorption_distance %10.4f scattering_distance %10.4f u_scattering %10.4f u_absorption %10.4f \n", 
      distance_to_boundary, absorption_distance, scattering_distance, u_scattering, u_absorption  ); 


    if (absorption_distance <= scattering_distance) 
    {   
        if (absorption_distance <= distance_to_boundary) 
        {   
            *time += absorption_distance/group_velocity ;   
            *position += absorption_distance*(*direction) ;

            float u_reemit = reemission_prob == 0.f ? 2.f : curand_uniform(&rng);  // avoid consumption at absorption when not scintillator

            if (u_reemit < reemission_prob)    
            {   
                *wavelength = scint_wavelength_hd20(rng);
                *direction = uniform_sphere(rng);
                *polarization = normalize(cross(uniform_sphere(rng), *direction));
                //flags.x = 0 ;   // no-boundary-yet for new direction TODO:elimate 

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
            //flags.x = 0 ;  // no-boundary-yet for new direction : TODO: eliminate 

            return CONTINUE;
        }       
          //  otherwise sail to boundary  
    }     // if scattering_distance < absorption_distance


    *position += distance_to_boundary*(*direction) ;
    *time += distance_to_boundary/group_velocity ;  

    return 0 ;
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

Changes:

* p.direction
* p.polarization

Consumes one random deciding between BOUNDARY_REFLECT and BOUNDARY_TRANSMIT

Returns: BOUNDARY_REFLECT or BOUNDARY_TRANSMIT

Notes:

* when geometry dictates TIR there is no dependence on u_reflect and always get reflection


**/

template <typename T>
inline QSIM_METHOD int qsim<T>::propagate_at_boundary(quad4& p, const qprd& prd, const qstate& s, curandStateXORWOW& rng)
{
    const float3& surface_normal = prd.normal ; 
    const float& n1 = s.material1.x ;
    const float& n2 = s.material2.x ;   
    float3* direction    = (float3*)&p.q1.f.x ; 
    float3* polarization = (float3*)&p.q2.f.x ; 

    const float eta = n1/n2 ; 
    const float c1 = -dot(*direction, surface_normal ); // c1 is flipped to be +ve  (G4 "cost1") when direction is against the normal,  1.f at normal incidence
    const bool normal_incidence = fabs(c1) > 0.999999f ; 

    const float eta_c1 = eta * c1 ; 

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law 
    bool tir = c2c2 < 0.f ; 


    const float EdotN = dot(*polarization, surface_normal ) ;  // used for TIR polarization

    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect

    //printf("//qsim.propagate_at_boundary n1 %10.4f n2 %10.4f eta %10.4f c1 %10.4f c2c2 %10.4f tir %d c2 %10.4f \n", n1, n2, eta, c1, c2c2, tir, c2 ); 

    const float n1c1 = n1*c1 ; 
    const float n2c2 = n2*c2 ; 
    const float n2c1 = n2*c1 ; 
    const float n1c2 = n1*c2 ; 

    //printf("//qsim.propagate_at_boundary c1 %10.4f n1c1 %10.4f n2c2 %10.4f n2c1 %10.4f n1c2 %10.4f \n", c1, n1c1, n2c2, n2c1, n1c2 ); 

    const float3 A_trans = normal_incidence ? *polarization : normalize(cross(*direction, surface_normal)) ;

    //printf("//qsim.propagate_at_boundary A_trans %10.4f %10.4f %10.4f  \n", A_trans.x, A_trans.y, A_trans.z ); 
    
    
    // decompose polarization onto incident orthogonal basis

    const float E1_perp = dot(*polarization, A_trans);   // fraction of E vector perpendicular to plane of incidence, ie S polarization
    const float3 E1pp = E1_perp * A_trans ;               // S-pol transverse component   
    const float3 E1pl = *polarization - E1pp ;           // P-pol parallel component 
    const float E1_parl = length(E1pl) ;

    //printf("//qsim.propagate_at_boundary E1pp ( %10.4f %10.4f %10.4f ) E1pl ( %10.4f %10.4f %10.4f ) E1_parl %10.4f \n", E1pp.x, E1pp.y, E1pp.z, E1pl.x, E1pl.y, E1pl.z, E1_parl ); 


    // G4OpBoundaryProcess at normal incidence, mentions Jackson and uses 
    //      A_trans  = OldPolarization; E1_perp = 0. E1_parl = 1. 
    // but that seems inconsistent with the above dot product, above is swapped cf that

    const float E2_perp_t = 2.f*n1c1*E1_perp/(n1c1+n2c2);  // Fresnel S-pol transmittance
    const float E2_parl_t = 2.f*n1c1*E1_parl/(n2c1+n1c2);  // Fresnel P-pol transmittance

    // SUSPECT DEVIATION FROM GEANT4 AT NORMAL INCIDENCE : SHOULD SET E1_perp 0.f E1_parl 1.f FOR NORMAL INCIDENCE 

    printf("//qsim.propagate_at_boundary E2_perp_t %10.4f E2_parl_t %10.4f \n", E2_perp_t, E2_parl_t ); 

    const float E2_perp_r = E2_perp_t - E1_perp;           // Fresnel S-pol reflectance
    const float E2_parl_r = (n2*E2_parl_t/n1) - E1_parl ;  // Fresnel P-pol reflectance


    const float2 E2_t = make_float2( E2_perp_t, E2_parl_t ) ; 
    const float2 E2_r = make_float2( E2_perp_r, E2_parl_r ) ; 

    const float  E2_total_t = dot(E2_t,E2_t) ; 


    const float2 TT = normalize(E2_t) ; 
    const float2 R = normalize(E2_r) ; 

    const float TransCoeff =  tir ? 0.0f : n2c2*E2_total_t/n1c1 ; 
    //  above 0.0f was until 2016/3/4 incorrectly a 1.0f 
    //  resulting in TIR yielding BT where BR is expected


    const float u_reflect = curand_uniform(&rng) ;
    bool reflect = u_reflect > TransCoeff  ;

    printf("//qsim.propagate_at_boundary n2c2 %10.4f E2_total_t %10.4f n1c1 %10.4f u_reflect %10.4f TransCoeff %10.4f (n2c2.E2_total_t/n1c1)  reflect %d \n", 
        n2c2,  E2_total_t, n1c1, u_reflect, TransCoeff, reflect ); 


    *direction = reflect
                    ?
                       *direction + 2.0f*c1*surface_normal
                    :
                       eta*(*direction) + (eta_c1 - c2)*surface_normal
                    ;

    const float3 A_paral = normalize(cross(*direction, A_trans));

    *polarization = reflect ?
                                ( tir ?
                                        -(*polarization) + 2.f*EdotN*surface_normal
                                      :
                                        R.x*A_trans + R.y*A_paral
                                )
                            :
                                TT.x*A_trans + TT.y*A_paral
                            ;

    //p.flags.i.x = 0 ;  // no-boundary-yet for new direction   TODO: eliminate 

    return reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ; 
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
HMM ? state struct to collect this thread locals ?
**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
{
    int gencode = gs.q0.i.x ; 
    //printf("//qsim.generate_photon gencode %d \n", gencode); 
    switch(gencode)
    {
        case OpticksGenstep_TORCH: generate_photon_torch(p, rng, gs, photon_id, genstep_id) ; break ; 
        default:                   generate_photon_dummy(p, rng, gs, photon_id, genstep_id) ; break ; 
    }
}


/**
qsim::generate_photon_torch
-----------------------------

Acting on the gensteps created eg in QEvent 

Contrast with CPU implementation sysrap SEvent::GenerateCenterExtentGenstepsPhotons

The gensteps are for example configured in SEvent::MakeCenterExtentGensteps

NB the sevent.h enum order is different to the python one  eg XYZ=0 


TODO: this has not been updated following the cxs torch genstep rejig with transforms into the genstep ?


**/

template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_torch(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
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


    // HMM:  photon_id is global to the launch 
    // but having a  local to the photons of this genstep index is more useful for this id
    // so the genstep needs to carry the photon_id offset in order convert into a local index

    unsigned char ucj = (photon_id < 255 ? photon_id : 255 ) ;
    gsid.c4.w = ucj ; 
    p.q3.u.w = gsid.u ;

} 



template <typename T>
inline QSIM_METHOD void qsim<T>::generate_photon_dummy(quad4& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id )
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

