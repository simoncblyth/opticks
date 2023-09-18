#pragma once
/**
qsim.h : GPU side struct prepared CPU side by QSim.hh
========================================================

qsim.h replaces the OptiX 6 context in a CUDA-centric way.
Canonical use is from CSGOptiX/CSGOptiX7.cu:simulate 

* qsim.h instance is uploaded once only at CSGOptiX instanciation 
  as this encompasses the physics not the event-by-event info.

* qsim encompasses global info relevant to all photons, meaning that any changes
  made to the qsim instance from single photon threads must be into thread-owned "idx" 
  slots into arrays to avoid interference 
 
* temporary working state local to each photon is held in *sctx* 
  and passed around using reference arguments

TODO:

1. get more of the below to work on CPU with mocked curand (and in future mocked tex2D and cudaTextureObject_t )
   NB must move decl and implementation to do this 

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
#include "sxyz.h"  
#include "sphoton.h"

#include "storch.h"
#include "scarrier.h"
#include "scurand.h"
#include "sevent.h"
#include "sstate.h"
#include "smatsur.h"


#ifndef PRODUCTION
#include "srec.h"
#include "sseq.h"
#include "stag.h"
#ifdef DEBUG_TAG
#define KLUDGE_FASTMATH_LOGF(u) (u < 0.998f ? __logf(u) : __logf(u) - 0.46735790f*1e-7f )
#endif
#endif

#include "sctx.h"

#include "qbase.h"
#include "qprop.h"
#include "qmultifilm.h"
#include "qbnd.h"
#include "qscint.h"
#include "qcerenkov.h"
#include "qpmt.h"
#include "tcomplex.h"



#if defined(MOCK_CUDA)
#else
struct curandStateXORWOW ; 
#endif

struct qcerenkov ; 

struct qsim
{
    qbase*              base ; 
    sevent*             evt ; 
    curandStateXORWOW*  rngstate ; 
    qbnd*               bnd ; 
    qmultifilm*         multifilm;
    qcerenkov*          cerenkov ; 
    qscint*             scint ; 
    qpmt<float>*        pmt ; 

    QSIM_METHOD void    generate_photon_dummy( sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD static float3 uniform_sphere(const float u0, const float u1); 
    QSIM_METHOD static float RandGaussQ_shoot( curandStateXORWOW& rng, float mean, float stdDev ); 
    QSIM_METHOD static void SmearNormal_SigmaAlpha( curandStateXORWOW& rng, float3* smeared_normal, const float3* direction, const float3* normal, float sigma_alpha, const sctx& ctx );
    QSIM_METHOD static void SmearNormal_Polish(     curandStateXORWOW& rng, float3* smeared_normal, const float3* direction, const float3* normal, float polish     , const sctx& ctx ); 

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
    QSIM_METHOD static float3 uniform_sphere(curandStateXORWOW& rng); 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) 
    QSIM_METHOD float4  multifilm_lookup(unsigned pmtType, unsigned boundary, float nm, float aoi);
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND )  || defined(MOCK_CUDA)
    QSIM_METHOD static void lambertian_direction(float3* dir, const float3* normal, float orient, curandStateXORWOW& rng, sctx& ctx ); 
    QSIM_METHOD static void random_direction_marsaglia(float3* dir, curandStateXORWOW& rng, sctx& ctx ); 
    QSIM_METHOD void rayleigh_scatter(curandStateXORWOW& rng, sctx& ctx ); 
    QSIM_METHOD int     propagate_to_boundary( unsigned& flag, curandStateXORWOW& rng, sctx& ctx ); 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
    QSIM_METHOD int     propagate_at_boundary(        unsigned& flag, curandStateXORWOW& rng, sctx& ctx, float theTransmittance=-1.f ) const ; 
    QSIM_METHOD int     propagate_at_boundary_with_T( unsigned& flag, curandStateXORWOW& rng, sctx& ctx, float theTransmittance ) const ; 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) 
    QSIM_METHOD int     propagate_at_multifilm(unsigned& flag, curandStateXORWOW& rng, sctx& ctx );
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
    QSIM_METHOD int     propagate_at_surface(           unsigned& flag, curandStateXORWOW& rng, sctx& ctx ); 
    QSIM_METHOD int     propagate_at_surface_Detect(    unsigned& flag, curandStateXORWOW& rng, sctx& ctx ) const ; 
#if defined( WITH_CUSTOM4 )
    QSIM_METHOD int     propagate_at_surface_CustomART( unsigned& flag, curandStateXORWOW& rng, sctx& ctx ) const ; 
#endif
#endif

#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
    QSIM_METHOD void    reflect_diffuse(                       curandStateXORWOW& rng, sctx& ctx );
    QSIM_METHOD void    reflect_specular(                      curandStateXORWOW& rng, sctx& ctx );

    QSIM_METHOD void    mock_propagate( sphoton& p, const quad2* mock_prd, curandStateXORWOW& rng, unsigned idx ); 
    QSIM_METHOD int     propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx ); 

    QSIM_METHOD void    hemisphere_polarized( unsigned polz, bool inwards, curandStateXORWOW& rng, sctx& ctx ); 
    QSIM_METHOD void    generate_photon_simtrace(   quad4&   p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
    QSIM_METHOD void    generate_photon(            sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const ; 
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
        scint(nullptr),
        pmt(nullptr)
    {
    }
#endif
}; 

inline QSIM_METHOD void qsim::generate_photon_dummy(sphoton& p_, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    quad4& p = (quad4&)p_ ; 
#ifndef PRODUCTION
    printf("//qsim::generate_photon_dummy  photon_id %3d genstep_id %3d  gs.q0.i ( gencode:%3d %3d %3d %3d ) \n", 
       photon_id, 
       genstep_id, 
       gs.q0.i.x, 
       gs.q0.i.y,
       gs.q0.i.z, 
       gs.q0.i.w
      );  
#endif
    p.q0.i.x = 1 ; p.q0.i.y = 2 ; p.q0.i.z = 3 ; p.q0.i.w = 4 ; 
    p.q1.i.x = 1 ; p.q1.i.y = 2 ; p.q1.i.z = 3 ; p.q1.i.w = 4 ; 
    p.q2.i.x = 1 ; p.q2.i.y = 2 ; p.q2.i.z = 3 ; p.q2.i.w = 4 ; 
    p.q3.i.x = 1 ; p.q3.i.y = 2 ; p.q3.i.z = 3 ; p.q3.i.w = 4 ; 

    p.set_flag(TORCH); 
}

inline QSIM_METHOD float3 qsim::uniform_sphere(const float u0, const float u1)
{
    float phi = u0*2.f*M_PIf;
    float cosTheta = 2.f*u1 - 1.f ; // -1.f -> 1.f 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);
    return make_float3(cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta); 
}


#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
/**
qsim::uniform_sphere
---------------------

**/
inline QSIM_METHOD float3 qsim::uniform_sphere(curandStateXORWOW& rng)
{
    float phi = curand_uniform(&rng)*2.f*M_PIf;
    float cosTheta = 2.f*curand_uniform(&rng) - 1.f ; // -1.f -> 1.f 
    float sinTheta = sqrtf(1.f-cosTheta*cosTheta);
    return make_float3(cosf(phi)*sinTheta, sinf(phi)*sinTheta, cosTheta); 
}

/**
qsim::RandGaussQ_shoot
------------------------

See:: 

    sysrap/tests/erfcinvf_Test.sh 
    sysrap/tests/S4MTRandGaussQTest.sh

    g4-cls RandGaussQ
    g4-cls G4MTRandGaussQ

**/
inline QSIM_METHOD float qsim::RandGaussQ_shoot( curandStateXORWOW& rng, float mean, float stdDev )
{
    float u2 = 2.f*curand_uniform(&rng) ;
    float v = -M_SQRT2f*erfcinvf(u2)*stdDev + mean ; 
    //printf("//qsim.RandGaussQ_shoot mean %10.5f stdDev %10.5f u2 %10.5f v %10.5f \n", mean, stdDev, u2, v  ) ; 
    return v ; 
}


/**
qsim::SmearNormal_SigmaAlpha
------------------------------

CAUTION : THIS CURRENTLY NOT USED BY ANYTHING OTHER THAN TESTS

The reason is that a Geant4 simulation with some sigma_alpha/polish 
surfaces will not actually use G4OpBoundaryProcess::GetFacetNormal unless
a bunch of conditions are satisfied such as having non-zero values of some 
of prob_sl/prob_ss/prob_bs that induces G4OpBoundaryProcess::ChooseReflection 
to return something other that LambertianReflection.

**THIS MAKES ME SUSPECT ACTUAL USE OF NORMAL SMEARING IS VERY RARE**

+----------------------+-------------------------------+------------------------------+----------------------+
| G4OpBoundaryProcess  | G4MaterialPropertiesIndex.hh  | G4MaterialPropertiesTable.cc | ChooseReflection     |  
+======================+===============================+==============================+======================+
| prob_sl              | kSPECULARLOBECONSTANT         | SPECULARLOBECONSTANT         | LobeReflection       | 
+----------------------+-------------------------------+------------------------------+----------------------+
| prob_ss              | kSPECULARSPIKECONSTANT        | SPECULARSPIKECONSTANT        | SpikeReflection      |  
+----------------------+-------------------------------+------------------------------+----------------------+
| prob_bs              | kBACKSCATTERCONSTANT          | BACKSCATTERCONSTANT          | BackScattering       |
+----------------------+-------------------------------+------------------------------+----------------------+
| all zero =>          |                               |                              | LambertianReflection |
+----------------------+-------------------------------+------------------------------+----------------------+

TODO: full simulation run with breakpoint "BP=C4OpBoundaryProcess::GetFacetNormal"

* C4 (not G4) as Custom4 is in use for the boundary process 

**/

inline QSIM_METHOD void qsim::SmearNormal_SigmaAlpha( 
    curandStateXORWOW& rng, 
    float3* smeared_normal, 
    const float3* direction, 
    const float3* normal, 
    float sigma_alpha, 
    const sctx& ctx
   )
{
#ifdef MOCK_CUDA_DEBUG
    bool dump = ctx.idx == -1 ; 
#endif

    if(sigma_alpha == 0.f)
    {
        *smeared_normal = *normal ;  
        return ; 
    } 
    float f_max = fminf(1.f,4.f*sigma_alpha);

#ifdef MOCK_CUDA_DEBUG
    if(dump) printf("//qsim::SmearNormal_SigmaAlpha.MOCK_CUDA_DEBUG sigma_alpha %10.5f f_max %10.5f  \n", sigma_alpha, f_max ); 
#endif

    float alpha, sin_alpha, phi, u0, u1, u2 ; 
    bool reject_alpha ; 
    bool reject_dir ; 

    do {
        do {
            //alpha = RandGaussQ_shoot(rng, 0.f, sigma_alpha );  // mean:0.f stdDev:sigma_alpha
            u0 = curand_uniform(&rng) ; 
            alpha = -M_SQRT2f*erfcinvf(2.f*u0)*sigma_alpha ; 

            sin_alpha = sinf(alpha); 
            u1 = curand_uniform(&rng) ; 
            reject_alpha = alpha >= M_PIf/2.f || (u1*f_max > sin_alpha) ; 

#ifdef MOCK_CUDA_DEBUG
            if(dump) printf("//qsim::SmearNormal_SigmaAlpha.MOCK_CUDA_DEBUG u0 %10.5f alpha %10.5f sin_alpha %10.5f u1 %10.5f u1*f_max %10.5f  (u1*f_max > sin_alpha) %d reject_alpha %d  \n", 
               u0, alpha, sin_alpha, u1, u1*f_max, (u1*f_max > sin_alpha), reject_alpha ); 
            // theres lots of alpha rejected : eg all -ve sin_alpha             
#endif

        } while( reject_alpha ) ; 

        u2 = curand_uniform(&rng) ; 
        phi = u2*M_PIf*2.f ;

        smeared_normal->x = sin_alpha * cosf(phi) ; 
        smeared_normal->y = sin_alpha * sinf(phi) ; 
        smeared_normal->z = cosf(alpha) ; 

        smath::rotateUz(*smeared_normal, *normal);
        reject_dir = dot(*smeared_normal, *direction ) >= 0.f ;   
        // reject smears that move the normal into same hemi as direction

#ifdef MOCK_CUDA_DEBUG
        if(dump) printf("//qsim::SmearNormal_SigmaAlpha.MOCK_CUDA_DEBUG u2 %10.5f phi %10.5f smeared_normal ( %10.5f, %10.5f, %10.5f)  reject_dir %d  \n", 
               u2, phi, smeared_normal->x, smeared_normal->y, smeared_normal->z, reject_dir ); 
#endif


    } while( reject_dir ) ; 
}

/**
qsim::SmearNormal_Polish
------------------------------

CAUTION : THIS CURRENTLY NOT USED BY ANYTHING OTHER THAN TESTS : SEE DETAILS ABOVE

**/

inline QSIM_METHOD void qsim::SmearNormal_Polish( 
    curandStateXORWOW& rng, 
    float3* smeared_normal, 
    const float3* direction, 
    const float3* normal, 
    float polish,
    const sctx& ctx
    )
{
#ifdef MOCK_CUDA_DEBUG
    bool dump = ctx.idx == -1 ; 
#endif

    if(polish == 1.f)
    {
        *smeared_normal = *normal ;  
        return ; 
    } 

    float u0, u1, u2 ; 
    float3 smear ;
    bool reject_mag ; 
    bool reject_dir ; 

    do {
        do {
            u0 = curand_uniform(&rng); 
            u1 = curand_uniform(&rng) ; 
            u2 = curand_uniform(&rng) ; 
            smear.x = 2.f*u0 - 1.f ; 
            smear.y = 2.f*u1 - 1.f ; 
            smear.z = 2.f*u2 - 1.f ; 
            reject_mag = length(smear) > 1.f  ;   // HMM: could this use just dot(smear, smear) ? 
       } 
       while( reject_mag );

       *smeared_normal = *normal + (1.f-polish)*smear;
       reject_dir = dot(*smeared_normal, *direction) >= 0.f ;  
    } 
    while( reject_dir );
    *smeared_normal = normalize(*smeared_normal); 
}





#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
/*
qsim::multifilm_lookup
-----------------------
 
*/

inline QSIM_METHOD float4 qsim::multifilm_lookup(unsigned pmtType, unsigned boundary, float nm, float aoi)
{
    float4 value = multifilm->lookup(pmtType, boundary, nm, aoi);
    return value;
}

#endif

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA) 

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
inline  QSIM_METHOD void qsim::lambertian_direction(float3* dir, const float3* normal, float orient, curandStateXORWOW& rng, sctx& ctx )
{
#ifdef DEBUG_PIDX
    int PIDX = 1 ;  // its static so cannot use base
    if(ctx.idx == PIDX )
    {
        printf("//qsim.lambertian_direction.head idx %d : normal = np.array([%10.5f,%10.5f,%10.5f]) ; orient = %10.5f  \n", 
            ctx.idx, normal->x, normal->y, normal->z, orient  );  
    }
#endif

    float ndotv ; 
    int count = 0 ; 
    float u ; 
    do
    {
        count++ ; 
        random_direction_marsaglia(dir, rng, ctx); // sets dir to random point on unit sphere 
        ndotv = dot( *dir, *normal )*orient ; 
        if( ndotv < 0.f )   
        {
            *dir = -1.f*(*dir) ; 
            ndotv = -1.f*ndotv ; 
        } 
        // when random dir is in opposite hemisphere to oriented normal 
        // flip the dir into same hemi and ndotv

        u = curand_uniform(&rng) ; 

#ifdef DEBUG_PIDX
        if(ctx.idx == PIDX)
        {
            printf("//qsim.lambertian_direction.loop idx %d : dir = np.array([%10.5f,%10.5f,%10.5f]) ; count = %d ; ndotv = %10.5f ; u = %10.5f \n", 
                ctx.idx, dir->x, dir->y, dir->z, count, ndotv, u   );  

        }
#endif
    } 
    while (!(u < ndotv) && (count < 1024)) ;  
    // distribution looks pretty similar without the while loop


#ifdef DEBUG_PIDX
    if(ctx.idx == PIDX)
    {
        printf("//qsim.lambertian_direction.tail idx %d : dir = np.array([%10.5f,%10.5f,%10.5f]) ; count = %d ; ndotv = %10.5f \n", 
            ctx.idx, dir->x, dir->y, dir->z, count, ndotv  );  

    }
#endif


}

/**
qsim::random_direction_marsaglia following G4RandomDirection
-------------------------------------------------------------

* https://mathworld.wolfram.com/SpherePointPicking.html

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

Marsaglia (1972) derived an elegant method that consists of picking u and v from independent 
uniform distributions on (-1,1) and rejecting points for which uu+vv >=1. 
So are picking points from within the (u,v) disc. 

For those remaining random points on the 2D (u,v) disc the below (u,v) to (x,y,z) 
mapping is used to give a 3D position on the unit sphere,

    x=2*u*sqrt(1-(uu+vv))	
    y=2*v*sqrt(1-(uu+vv))	
    z=1-2(uu+vv)

Checking normalization, it reduces to 1::

   xx + yy + zz = 
         4uu (1-(uu+vv)) 
         4vv (1-(uu+vv)) +
        1 -4(uu+vv) + 4(uu+vv)(uu+vv)   
                = 1 

So that means the random 3D (x,y,z) points are on the unit sphere. 

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


inline QSIM_METHOD void qsim::random_direction_marsaglia(float3* dir,  curandStateXORWOW& rng, sctx& ctx  )
{
    // NB: no use of ctx.tagr so this has not been random aligned 
    float u0, u1 ; 
    float u, v, b, a  ; 
    do 
    {
        u0 = curand_uniform(&rng);
        u1 = curand_uniform(&rng);
        //if( idx == 0u ) printf("//qsim.random_direction_marsaglia idx %d u0 %10.4f u1 %10.4f \n", ctx.idx, u0, u1 ); 
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

inline QSIM_METHOD void qsim::rayleigh_scatter(curandStateXORWOW& rng, sctx& ctx ) 
{
    sphoton& p = ctx.p ; 
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

#ifdef DEBUG_TAG
        stagr& tagr = ctx.tagr ;  // UNTESTED
        tagr.add(stag_sc, u0); 
        tagr.add(stag_sc, u1); 
        tagr.add(stag_sc, u2); 
        tagr.add(stag_sc, u3); 
        tagr.add(stag_sc, u4); 
#endif
        float cosTheta = u0 ;
        float sinTheta = sqrtf(1.0f-u0*u0);
        if(u1 < 0.5f ) cosTheta = -cosTheta ; 
        // could use uniform_sphere here : but not doing so to follow G4OpRayleigh more closely 


        float sinPhi ; 
        float cosPhi ; 
#if defined(MOCK_CURAND ) || defined(MOCK_CUDA)
        __sincosf(2.f*M_PIf*u2,&sinPhi,&cosPhi);
#else
        sincosf(2.f*M_PIf*u2,&sinPhi,&cosPhi);
#endif
       
        direction.x = sinTheta * cosPhi;
        direction.y = sinTheta * sinPhi;
        direction.z = cosTheta ;

        smath::rotateUz(direction, p.mom );

        float constant = -dot(direction, p.pol ); 

        polarization.x = p.pol.x + constant*direction.x ;
        polarization.y = p.pol.y + constant*direction.y ;
        polarization.z = p.pol.z + constant*direction.z ;

        if(dot(polarization, polarization) == 0.f )
        {

#if defined( MOCK_CURAND ) || defined(MOCK_CUDA)
            __sincosf(2.f*M_PIf*u3,&sinPhi,&cosPhi);
#else
            sincosf(2.f*M_PIf*u3,&sinPhi,&cosPhi);
#endif

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
        float doCosTheta = dot(polarization, p.pol ) ;
        float doCosTheta2 = doCosTheta*doCosTheta ;
        looping = doCosTheta2 < u4 ;

    } while ( looping ) ;

    p.mom = direction ;
    p.pol = polarization ;
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


TODO: 
   whilst in measurement iteration try changing the four "return" 
   in the below into a single return: by setting command variable 
   and returning that 

**/



inline QSIM_METHOD int qsim::propagate_to_boundary(unsigned& flag, curandStateXORWOW& rng, sctx& ctx)
{
    sphoton& p = ctx.p ; 
    const sstate& s = ctx.s ; 

    const float& absorption_length = s.material1.y ; 
    const float& scattering_length = s.material1.z ; 
    const float& reemission_prob = s.material1.w ; 
    const float& group_velocity = s.m1group2.x ; 
    const float& distance_to_boundary = ctx.prd->q0.f.w ; 


#ifdef DEBUG_TAG
    float u_to_sci = curand_uniform(&rng) ;  // purely for alignment with G4 
    float u_to_bnd = curand_uniform(&rng) ;  // purely for alignment with G4 
#endif
    float u_scattering = curand_uniform(&rng) ;
    float u_absorption = curand_uniform(&rng) ;

#ifdef DEBUG_TAG
    stagr& tagr = ctx.tagr ; 
    tagr.add( stag_to_sci, u_to_sci); 
    tagr.add( stag_to_bnd, u_to_bnd); 
    tagr.add( stag_to_sca, u_scattering); 
    tagr.add( stag_to_abs, u_absorption); 

    // see notes/issues/U4LogTest_maybe_replacing_G4Log_G4UniformRand_in_Absorption_and_Scattering_with_float_version_will_avoid_deviations.rst
    float scattering_distance = -scattering_length*KLUDGE_FASTMATH_LOGF(u_scattering);   
    float absorption_distance = -absorption_length*KLUDGE_FASTMATH_LOGF(u_absorption);
#else
    float scattering_distance = -scattering_length*logf(u_scattering);   
    float absorption_distance = -absorption_length*logf(u_absorption);
#endif

#ifdef DEBUG_PIDX

    if(ctx.idx == base->pidx)
    {
    printf("//qsim.propagate_to_boundary.head idx %d : u_absorption %10.8f logf(u_absorption) %10.8f absorption_length %10.4f absorption_distance %10.6f \n", 
        ctx.idx, u_absorption, logf(u_absorption), absorption_length, absorption_distance );         

    printf("//qsim.propagate_to_boundary.head idx %d : post = np.array([%10.5f,%10.5f,%10.5f,%10.5f]) \n", 
        ctx.idx, p.pos.x, p.pos.y, p.pos.z, p.time );  

    printf("//qsim.propagate_to_boundary.head idx %d : distance_to_boundary %10.4f absorption_distance %10.4f scattering_distance %10.4f \n",
             ctx.idx, distance_to_boundary, absorption_distance, scattering_distance );

    printf("//qsim.propagate_to_boundary.head idx %d : u_scattering %10.4f u_absorption %10.4f \n", 
             ctx.idx, u_scattering, u_absorption  ); 

    }
#endif



 

    if (absorption_distance <= scattering_distance) 
    {   
        if (absorption_distance <= distance_to_boundary) 
        {   
            p.time += absorption_distance/group_velocity ;   
            p.pos  += absorption_distance*(p.mom) ;

#ifdef DEBUG_PIDX
            float absorb_time_delta = absorption_distance/group_velocity ; 
            if( ctx.idx == base->pidx ) 
            {
            printf("//qsim.propagate_to_boundary.body.BULK_ABSORB idx %d : post = np.array([%10.5f,%10.5f,%10.5f,%10.5f]) ; absorb_time_delta = %10.8f   \n", 
                    ctx.idx, p.pos.x, p.pos.y, p.pos.z, p.time, absorb_time_delta  );  

            }
#endif

            float u_reemit = reemission_prob == 0.f ? 2.f : curand_uniform(&rng);  // avoid consumption at absorption when not scintillator

#ifdef DEBUG_TAG
            if( u_reemit != 2.f ) tagr.add( stag_to_ree, u_reemit) ; 
#endif


            if (u_reemit < reemission_prob)    
            {  
                float u_re_wavelength = curand_uniform(&rng);  
                float u_re_mom_ph = curand_uniform(&rng); 
                float u_re_mom_ct = curand_uniform(&rng); 
                float u_re_pol_ph = curand_uniform(&rng); 
                float u_re_pol_ct = curand_uniform(&rng); 

                p.wavelength = scint->wavelength(u_re_wavelength);
                p.mom = uniform_sphere(u_re_mom_ph, u_re_mom_ct);
                p.pol = normalize(cross(uniform_sphere(u_re_pol_ph, u_re_pol_ct), p.mom));

#ifdef DEBUG_TAG
                tagr.add( stag_re_wl, u_re_wavelength); 
                tagr.add( stag_re_mom_ph, u_re_mom_ph); 
                tagr.add( stag_re_mom_ct, u_re_mom_ct); 
                tagr.add( stag_re_pol_ph, u_re_pol_ph); 
                tagr.add( stag_re_pol_ct, u_re_pol_ct); 
#endif

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

            rayleigh_scatter(rng, ctx);  // changes dir and pol, consumes 5u at each turn of rejection sampling loop

            flag = BULK_SCATTER;

            return CONTINUE;
        }       
          //  otherwise sail to boundary  
    }     // if scattering_distance < absorption_distance



    p.pos  += distance_to_boundary*(p.mom) ;
    p.time += distance_to_boundary/group_velocity   ;  

#ifdef DEBUG_PIDX
    float sail_time_delta = distance_to_boundary/group_velocity ; 
    if( ctx.idx == base->pidx ) printf("//qsim.propagate_to_boundary.tail.SAIL idx %d : post = np.array([%10.5f,%10.5f,%10.5f,%10.5f]) ;  sail_time_delta = %10.5f   \n", 
          ctx.idx, p.pos.x, p.pos.y, p.pos.z, p.time, sail_time_delta  );  
#endif

    return BOUNDARY ;
}
#endif
#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
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

+---------------------+------------------+-------------------------------+----------+
| output flag         |   command        |  changed                      |  note    |
+=====================+==================+===============================+==========+
|   BOUNDARY_REFLECT  |    -             |  direction, polarization      |          |
+---------------------+------------------+-------------------------------+----------+
|   BOUNDARY_TRANSMIT |    -             |  direction, polarization      |          |
+---------------------+------------------+-------------------------------+----------+

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
This definition is used by *fill_state* in order to determine properties 
of this material m1 and the next material m2 on the other side of the boundary.   

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



**Normal Incidence Special Case**
 
Judging normal_incidence based on absolete dot product being exactly unity "c1 == 1.f" is problematic 
as when very near to normal incidence there are vectors for which the absolute dot product 
is not quite 1.f but the cross product does give an exactly zero vector which gives 
A_trans (nan, nan, nan) from the normalize doing : (zero,zero,zero)/zero.   

Solution is to judge normal incidence based on trans_length as that is what the 
calculation actually needs to be non-zero in order to be able to normalize trans to give A_trans.

However using "bool normal_incidence = trans_length == 0.f" also problematic
as it means would be using very small trans vectors to define A_trans and this
would cause a difference with double precision Geant4 and float precision Opticks. 
So try using a cutoff "trans_length < 1e-6f" below which to special case normal 
incidence. 

**/

inline QSIM_METHOD int qsim::propagate_at_boundary(unsigned& flag, curandStateXORWOW& rng, sctx& ctx, float theTransmittance ) const 
{
    sphoton& p = ctx.p ; 
    const sstate& s = ctx.s ; 

    const float& n1 = s.material1.x ;
    const float& n2 = s.material2.x ;   
    const float eta = n1/n2 ; 

    const float3* normal = (float3*)&ctx.prd->q0.f.x ;     // geometrical outwards normal 

    const float _c1 = -dot(p.mom, *normal );                // _c1 : cos(angle_of_incidence) not yet oriented
    const float3 oriented_normal = _c1 < 0.f ? -(*normal) : (*normal) ; // oriented against incident p.mom
    const float3 trans = cross(p.mom, oriented_normal) ;   // perpendicular to plane of incidence, S-pol direction 
    const float trans_length = length(trans) ;             // same as sin(theta), as p.mom and oriented_normal are unit vectors
    const bool normal_incidence = trans_length < 1e-6f  ;  // p.mom parallel/anti-parallel to oriented_normal  
    const float3 A_trans = normal_incidence ? p.pol : trans/trans_length ; // normalized unit vector : perpendicular to plane of incidence
    const float E1_perp = dot(p.pol, A_trans);     // amplitude of polarization in direction perpendicular to plane of incidence, ie S polarization
 
    const float c1 = fabs(_c1) ; 

#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.propagate_at_boundary.head idx %d : theTransmittance = %10.8f \n", ctx.idx, theTransmittance  ); 
    printf("//qsim.propagate_at_boundary.head idx %d : nrm = np.array([%10.8f,%10.8f,%10.8f]) ; lnrm = %10.8f  \n", 
         ctx.idx, oriented_normal.x, oriented_normal.y, oriented_normal.z, length(oriented_normal) ); 
    printf("//qsim.propagate_at_boundary.head idx %d : pos = np.array([%10.5f,%10.5f,%10.5f]) ; lpos = %10.8f \n", 
         ctx.idx, p.pos.x, p.pos.y, p.pos.z, length(p.pos) ); 
    printf("//qsim.propagate_at_boundary.head idx %d : mom0 = np.array([%10.8f,%10.8f,%10.8f]) ; lmom0 = %10.8f \n", 
         ctx.idx, p.mom.x, p.mom.y, p.mom.z, length(p.mom)  ); 
    printf("//qsim.propagate_at_boundary.head idx %d : pol0 = np.array([%10.8f,%10.8f,%10.8f]) ; lpol0 = %10.8f \n",
          ctx.idx, p.pol.x, p.pol.y, p.pol.z, length(p.pol)  );
    printf("//qsim.propagate_at_boundary.head idx %d : n1,n2,eta = (%10.8f,%10.8f,%10.8f) \n", ctx.idx, n1, n2, eta ); 
    printf("//qsim.propagate_at_boundary.head idx %d : c1 = %10.8f ; normal_incidence = %d \n", ctx.idx, c1, normal_incidence ); 
    }
#endif

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law and trig identity 
    bool tir = c2c2 < 0.f ; 
    const float EdotN = dot(p.pol, oriented_normal ) ;  // used for TIR polarization
    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect
    const float n1c1 = n1*c1 ;
    const float n2c2 = n2*c2 ; 
    const float n2c1 = n2*c1 ; 
    const float n1c2 = n1*c2 ; 

    const float2 E1   = normal_incidence ? make_float2( 0.f, 1.f) : make_float2( E1_perp , length( p.pol - (E1_perp*A_trans) ) ); 
    const float2 E2_t = make_float2(  2.f*n1c1*E1.x/(n1c1+n2c2), 2.f*n1c1*E1.y/(n2c1+n1c2) ) ;  // ( S:perp, P:parl )  
    const float2 E2_r = make_float2( E2_t.x - E1.x             , (n2*E2_t.y/n1) - E1.y     ) ;  // ( S:perp, P:parl )    
    const float2 RR = normalize(E2_r) ; 
    const float2 TT = normalize(E2_t) ; 
    const float TransCoeff = theTransmittance >= 0.f ? 
                                                           theTransmittance 
                                                     :
                                                           ( tir || n1c1 == 0.f ? 0.f : n2c2*dot(E2_t,E2_t)/n1c1 ) 
                                                     ; 

    /* 
    E1, E2_t, E2_t: incident, transmitted and reflected amplitudes in S and P directions 
    RR, TT: normalized  
    */

#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.propagate_at_boundary.body idx %d : TransCoeff = %10.8f ; n1c1 = %10.8f ; n2c2 = %10.8f \n",
            ctx.idx, TransCoeff, n1c1, n2c2 ); 

    printf("//qsim.propagate_at_boundary.body idx %d : E2_t = np.array([%10.8f,%10.8f]) ; lE2_t = %10.8f \n",
            ctx.idx,  E2_t.x, E2_t.y, length(E2_t) );

    printf("//qsim.propagate_at_boundary.body idx %d : A_trans = np.array([%10.8f,%10.8f,%10.8f]) ; lA_trans = %10.8f \n",
            ctx.idx,  A_trans.x, A_trans.y, A_trans.z, length(A_trans) ); 

    }
#endif


#ifdef DEBUG_TAG
    const float u_boundary_burn = curand_uniform(&rng) ;  // needed for random consumption alignment with Geant4 G4OpBoundaryProcess::PostStepDoIt
#endif
    const float u_reflect = curand_uniform(&rng) ;
    bool reflect = u_reflect > TransCoeff  ;

#ifdef DEBUG_TAG
    stagr& tagr = ctx.tagr ; 
    tagr.add( stag_at_burn_sf_sd, u_boundary_burn); 
    tagr.add( stag_at_ref,  u_reflect); 
#endif

#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.propagate_at_boundary.body idx %d : u_reflect %10.4f TransCoeff %10.4f reflect %d \n", 
              ctx.idx,  u_reflect, TransCoeff, reflect   ); 

    printf("//qsim.propagate_at_boundary.body idx %d : mom0 = np.array([%10.8f,%10.8f,%10.8f]) ; lmom0 = %10.8f \n", 
               ctx.idx, p.mom.x, p.mom.y, p.mom.z, length(p.mom) ) ; 

    printf("//qsim.propagate_at_boundary.body idx %d : pos = np.array([%10.5f,%10.5f,%10.5f]) ; lpos = %10.8f \n", 
               ctx.idx, p.pos.x, p.pos.y, p.pos.z, length(p.pos)  ); 

    printf("//qsim.propagate_at_boundary.body idx %d : nrm = np.array([%10.8f,%10.8f,%10.8f]) ; lnrm = %10.8f \n", 
               ctx.idx, oriented_normal.x, oriented_normal.y, oriented_normal.z, length(oriented_normal) ) ; 

    printf("//qsim.propagate_at_boundary.body idx %d : n1 = %10.8f ; n2 = %10.8f ; eta = %10.8f  \n",
               ctx.idx, n1, n2, eta );  

    printf("//qsim.propagate_at_boundary.body idx %d : c1 = %10.8f ; eta_c1 = %10.8f ; c2 = %10.8f ; eta_c1__c2 = %10.8f \n",
               ctx.idx, c1, eta*c1, c2, (eta*c1 - c2) );  

    }
#endif 

    p.mom = reflect
                    ?
                       p.mom + 2.0f*c1*oriented_normal
                    :
                       eta*(p.mom) + (eta*c1 - c2)*oriented_normal
                    ;


    // Q: Does the new p.mom need to be normalized ?
    // A: NO, it is inherently normalized as derived in the comment below


    const float3 A_paral = normalize(cross(p.mom, A_trans));   // new P-pol direction 

    p.pol =  normal_incidence ?
                                         ( reflect ?  p.pol*(n2>n1? -1.f:1.f) : p.pol )
                                      : 
                                         ( reflect ?
                                                   ( tir ?  -p.pol + 2.f*EdotN*oriented_normal : RR.x*A_trans + RR.y*A_paral )

                                                   :
                                                       TT.x*A_trans + TT.y*A_paral 
                                             
                                                   )
                                      ;



     // Q: Above expression kinda implies A_trans and A_paral are same for reflect and transmit ?
     // A: NO IT DOESNT,
     //    A_trans is the same (except for normal incidence) as there is only one perpendicular 
     //    to the plane of incidence which is the same for i,r,t.
     //
     //    A_paral depends on the new p.mom (is has to be orthogonal to p.mom and A_trans) 
     //    and p.mom of course is different for r and t 
     //    (the reflect bool is used in multiple places, not just here)
  


#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.propagate_at_boundary.tail idx %d : reflect %d tir %d TransCoeff %10.4f u_reflect %10.4f \n", ctx.idx, reflect, tir, TransCoeff, u_reflect );  
    printf("//qsim.propagate_at_boundary.tail idx %d : mom1 = np.array([%10.8f,%10.8f,%10.8f]) ; lmom1 = %10.8f  \n", 
        ctx.idx, p.mom.x, p.mom.y, p.mom.z, length(p.mom) ); 
    printf("//qsim.propagate_at_boundary.tail idx %d : pol1 = np.array([%10.8f,%10.8f,%10.8f]) ; lpol1 = %10.8f \n", 
        ctx.idx, p.pol.x, p.pol.y, p.pol.z, length(p.pol) ); 
    }
#endif

    /*
    if(ctx.idx == 251959)
    {
        printf("//qsim.propagate_at_boundary RR.x %10.4f A_trans (%10.4f %10.4f %10.4f )  RR.y %10.4f  A_paral (%10.4f %10.4f %10.4f ) \n", 
              RR.x, A_trans.x, A_trans.y, A_trans.z,
              RR.y, A_paral.x, A_paral.y, A_paral.z ); 

        printf("//qsim.propagate_at_boundary reflect %d  tir %d polarization (%10.4f, %10.4f, %10.4f) \n", reflect, tir, p.pol.x, p.pol.y, p.pol.z );  
    }
    */

    flag = reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ; 


#ifdef DEBUG_TAG
    if( flag ==  BOUNDARY_REFLECT )
    {
        const float u_br_align_0 = curand_uniform(&rng) ;  
        const float u_br_align_1 = curand_uniform(&rng) ;
        const float u_br_align_2 = curand_uniform(&rng) ;
        const float u_br_align_3 = curand_uniform(&rng) ;

        // switched below standard tags from stag_br_align_0/1/2/3 to simplify A:tag to B:stack mapping 
        tagr.add( stag_to_sci, u_br_align_0 );
        tagr.add( stag_to_bnd, u_br_align_1 ); 
        tagr.add( stag_to_sca, u_br_align_2 ); 
        tagr.add( stag_to_abs, u_br_align_3 ); 
    }
#endif

    return CONTINUE ; 
}
/**
qsim::propagate_at_boundary_with_T
------------------------------------

Variant of qsim::propagate_at_boundary where a value of theTransmittance 
is passed in as an argument (eg from CustomART calculation) rather then 
calculated here. 

HMM: was hoping that passing in theTransmittance would afford some 
simplifications, but it seems almost no simplification is possible 
as need to calculate everything anyhow for the polarization. 

Given that this is almost exactly the same as qsim::propagate_at_boundary
and no simplification is afforded, might as well just keep using that
and leave this just for testing. 

**/

inline QSIM_METHOD int qsim::propagate_at_boundary_with_T(unsigned& flag, curandStateXORWOW& rng, sctx& ctx, float theTransmittance ) const 
{
    sphoton& p = ctx.p ; 
    const sstate& s = ctx.s ; 

    const float& n1 = s.material1.x ;
    const float& n2 = s.material2.x ;   
    const float eta = n1/n2 ; 

    const float3* normal = (float3*)&ctx.prd->q0.f.x ;     // geometrical outwards normal 

    const float _c1 = -dot(p.mom, *normal );                // _c1 : cos(angle_of_incidence) not yet oriented
    const float3 oriented_normal = _c1 < 0.f ? -(*normal) : (*normal) ; // oriented against incident p.mom
    const float3 trans = cross(p.mom, oriented_normal) ;   // perpendicular to plane of incidence, S-pol direction 
    const float trans_length = length(trans) ;             // same as sin(theta), as p.mom and oriented_normal are unit vectors
    const bool normal_incidence = trans_length < 1e-6f  ;  // p.mom parallel/anti-parallel to oriented_normal  
    const float3 A_trans = normal_incidence ? p.pol : trans/trans_length ; // normalized unit vector : perpendicular to plane of incidence
    const float E1_perp = dot(p.pol, A_trans);     // amplitude of polarization in direction perpendicular to plane of incidence, ie S polarization
 
    const float c1 = fabs(_c1) ; 

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law and trig identity 
    bool tir = c2c2 < 0.f ; 
    const float EdotN = dot(p.pol, oriented_normal ) ;  // used for TIR polarization
    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f : so will always reflect
    const float n1c1 = n1*c1 ;
    const float n2c2 = n2*c2 ; 
    const float n2c1 = n2*c1 ; 
    const float n1c2 = n1*c2 ; 

    const float2 E1   = normal_incidence ? make_float2( 0.f, 1.f) : make_float2( E1_perp , length( p.pol - (E1_perp*A_trans) ) ); 
    const float2 E2_t = make_float2(  2.f*n1c1*E1.x/(n1c1+n2c2), 2.f*n1c1*E1.y/(n2c1+n1c2) ) ;  // ( S:perp, P:parl )  
    const float2 E2_r = make_float2( E2_t.x - E1.x             , (n2*E2_t.y/n1) - E1.y     ) ;  // ( S:perp, P:parl )    
    const float2 RR = normalize(E2_r) ; 
    const float2 TT = normalize(E2_t) ; 


/*
    const float TransCoeff = theTransmittance >= 0.f ? 
                                                           theTransmittance 
                                                     :
                                                           ( tir || n1c1 == 0.f ? 0.f : n2c2*dot(E2_t,E2_t)/n1c1 ) 
                                                     ; 
*/

    const float& TransCoeff = theTransmittance ; 

    const float u_reflect = curand_uniform(&rng) ;
    bool reflect = u_reflect > TransCoeff  ;

    p.mom = reflect
                    ?
                       p.mom + 2.0f*c1*oriented_normal
                    :
                       eta*(p.mom) + (eta*c1 - c2)*oriented_normal
                    ;


    const float3 A_paral = normalize(cross(p.mom, A_trans));   // new P-pol direction 

    p.pol =  normal_incidence ?
                                         ( reflect ?  p.pol*(n2>n1? -1.f:1.f) : p.pol )
                                      : 
                                         ( reflect ?
                                                   ( tir ?  -p.pol + 2.f*EdotN*oriented_normal : RR.x*A_trans + RR.y*A_paral )

                                                   :
                                                       TT.x*A_trans + TT.y*A_paral 
                                             
                                                   )
                                      ;

    flag = reflect ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ; 


    return CONTINUE ; 
}

#endif








/**
Reflected momentum vector 
---------------------------

::

     p.mom(new) = p.mom(old) + 2.f*c1*oriented_normal


     PdotN = dot(p.mom(old), oriented_normal) = -c1      

     "c1" is +ve, above dot product -ve as incident vector is against the normal 

     p.mom(new) = p.mom(old) -2.f dot(p.mom(old), oriented_normal) * oriented_normal 

     NewMomentum = OldMomentum - 2*PdotN*oriented_normal 



     OA = oriented_normal

     IO = OC = p.mom (old)
          OR = p.mom (new)

          OR = OC + CD + DR  = OC + 2.f*DR 
          CD = DR = c1*oriented_normal 


                  s1   s1 
                I----A----R
                .\   .   /.
            c1  . \  .  / .  c1
                .  \ . /  .
                .   \./   .
       ---------+----O----D------
                :    :.   :
                :    : .  :  c1
           c1   :    :  . :  
                :    :   .:
                +....B....C
                 s1    s1 


* NB that is inherently normalized, as demonstrated in the below reference 


Transmitted momentum vector
-------------------------------

* https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
* ~/opticks_refs/bram_de_greve_reflection_refraction_in_ray_tracing.pdf


::  

     eta = n1/n2 

     p.mom(new) =  eta*p.mom(old) + (eta*c1 - c2)*oriented_normal


Consider the perp and para components of the incident and transmitted mom vectors::

    i = i_perp + i_para 
    t = t_perp + t_para

    i_perp = ( i.n ) n      (component of i in normal direction)
    t_perp = ( t.n ) n      (component of t in normal direction )

    i_para = i - (i.n) n    
    t_para = t - (t.n) n    

    # HMM caution with backwards oriented normal direction convention
    #     and dot product signs 


Snell::

    n1 s1 = n2 s2 

    eta s1 = s2      (eta = n1/n2)   
        
    s2 = eta s1 

    |i_para| = s1 
    |t_para| = s2 

    t_para  = eta i_para     (as they point in same direction)

    t_para  = eta ( i - (i.n) n ) 
            = eta ( i + c1 n )     

    t_perp  = - sqrt( 1-|t_para|^2 ) n     (-ve as opposite direction to oriented normal)
            = - c2 n 

    t   = eta i + ( eta c1 - c2 ) n        


Now is that inherently normalized ? YES

   t.t =  eta*eta*i.i  + (eta c1 - c2 )*(eta c1 - c2 )*n.n + 2 eta (eta c1 - c2) i.n      

         (i.i = 1, n.n = 1, i.n = -c1 )

       = eta*eta + (eta c1 - c2)*(eta c1 - c2) + 2 eta(eta c1 - c2) (-c1 )

       = eta eta  +  eta eta c1 c1 + c2 c2 - 2 eta c1 c2  - 2 eta eta c1 c1 + 2 eta c2 c1   
                                           --------------                    --------------
       =   eta eta  ( 1 - c1 c1 ) + c2 c2

       = 1.f      ( form above   c2c2 = 1.f - eta eta ( 1.f - c1c1 ) )    # snell and trig identity 



Compare transmitted vector with G4OpBoundaryProcess::DielectricDielectric
----------------------------------------------------------------------------

::

    1226                 theStatus = FresnelRefraction;
    1227 
    1228                 if (sint1 > 0.0) {      // incident ray oblique
    1229 
    1230                    alpha = cost1 - cost2*(Rindex2/Rindex1);
    1231                    NewMomentum = OldMomentum + alpha*theFacetNormal;
    1232                    NewMomentum = NewMomentum.unit();


    t = eta i  + (eta c1 - c2 ) n      eta = n1/n2 
    t/eta = i + (c1 - c2/eta ) n 

    Because Geant4 normalizes NewMomentum it gets away with 
    playing fast and loose with factors of 1/eta 


**/







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


/*
qsim::propagate_at_multifilm
-------------------------------

based on https://juno.ihep.ac.cn/trac/browser/offline/trunk/Simulation/DetSimV2/PMTSim/src/junoPMTOpticalModel.cc 
which follows a flawed approach to polarization 

Rs: s-component reflect probability
Ts: s-component transmit probability
Rp: p-component reflect probability
Tp: p-component reflect probability

TODO:
   access the qe
   access the multifilm catagory
   optical photon polarization

*/

#if defined(__CUDACC__) || defined(__CUDABE__) 
inline QSIM_METHOD int qsim::propagate_at_multifilm(unsigned& flag, curandStateXORWOW& rng, sctx& ctx )
{ 
    sphoton& p = ctx.p ; 
    sstate& s = ctx.s ;  

    const float& n1 = s.material1.x ; 
    const float& n2 = s.material2.x ;    
    const float eta = n1/n2 ;  

    const float3* normal = ctx.prd->normal() ;
    const float _c1 = dot(p.mom,(*normal));
    const float3 oriented_normal = _c1 < 0.f ? (*normal) : -(*normal) ;

    const float c1 = fabs(_c1) ;  
    const float s1 = sqrtf(1.f-c1*c1);

    float EsEs = s1 > 0.f ? dot(p.pol, cross( p.mom, *normal))/s1 : 0. ;
    EsEs *= EsEs;   //   orienting normal doesnt matter as squared : this is S_vs_P power fraction


    unsigned boundaryIdx = _c1 < 0.f ? 0u : 1u ; // if _c1 < 0 , thus the photon position is in glass ( kInGlass = 0 ) . 
    unsigned pmtType = 0u ;  //Fix me the pmtType need to find  
    float wavelength = p.wavelength ; 
    //boundaryIdx = 1u ;//just use to test
     
    float4 RsTsRpTp = multifilm->lookup(pmtType, boundaryIdx, wavelength, c1);
   
    float3 ART ;
    ART.z = RsTsRpTp.y*EsEs + RsTsRpTp.w*(1.f - EsEs);
    ART.y = RsTsRpTp.x*EsEs + RsTsRpTp.z*(1.f - EsEs);
    ART.x = 1.f - (ART.y+ART.z);

    const float& A = ART.x ; 
    const float& R = ART.y ; 
    //const float& T = ART.z ; 


    float4 RsTsRpTpNormal = multifilm->lookup(pmtType, 0u, wavelength, 1.f ); 
     // photon is in glass, aoi = 90deg cos_theta = 1.f

    float3 ART_normal;
    ART_normal.z = 0.5f*(RsTsRpTpNormal.y + RsTsRpTpNormal.w); // T:0.5f*(Ts+Tp)
    ART_normal.y = 0.5f*(RsTsRpTpNormal.x + RsTsRpTpNormal.z); // R:0.5f*(Rs+Rp)
    ART_normal.x = 1.f -(ART_normal.y + ART_normal.z) ;        // 1.f - (R+T) 

    const float& An = ART_normal.x ;  
    const float _qe = 0.0f;   // TODO: need to get _qe 
    const float D = _qe/An ;

    /* 
    HUH: why the complex math here ?  s1, eta are real so s2,c2 are real also 

    cuComplex s2 = make_cuComplex(s1*eta, 0.f) ;
    cuComplex s2s2 = cuCmulf(s2,s2);

    cuComplex one= make_cuComplex (1.f , 0.f) ;
    cuComplex c2 = tcomplex::cuSqrtf( cuCsubf( one,s2s2) );
    */

    const float c2c2 = 1.f - eta*eta*(1.f - c1 * c1 ) ;   // Snells law and trig identity 
    bool tir = c2c2 < 0.f ; 
    const float c2 = tir ? 0.f : sqrtf(c2c2) ;   // c2 chosen +ve, set to 0.f for TIR => reflection_coefficient = 1.0f 

    // TODO: how to handle TIR, does the TMM stack calc already handle that ? 

    const float u0 = curand_uniform(&rng) ;
    const float u1 = curand_uniform(&rng) ;

    flag =  u0 < A ? 
                      ( u1 < D ? SURFACE_DETECT : SURFACE_ABSORB )
                   : 
                      ( u0 < A + R ? BOUNDARY_REFLECT : BOUNDARY_TRANSMIT ) 
                   ; 



    if(flag == BOUNDARY_REFLECT)
    {
        p.mom -= 2.f * dot( p.mom, oriented_normal)*oriented_normal ;
        p.pol -= 2.f * dot( p.pol, oriented_normal)*oriented_normal ; // TODO: WHERE DOES THIS COME FROM ?
    }
    else if( flag == BOUNDARY_TRANSMIT )
    {
        p.mom = eta*p.mom + (eta*c1 - c2)*oriented_normal ;   // inherently normalized, see qsim.h:propagate_at_boundary
        p.pol = normalize((p.pol - dot(p.pol,p.mom)*p.mom));  // TODO: WHERE DOES THIS COME FROM ?
    }

    // TODO: compare these pol with propagate_at_boundary : maybe pull out some common code 

    return ( flag == SURFACE_DETECT || flag == SURFACE_ABSORB ) ? BREAK : CONTINUE ; 
}

#endif


#if defined(__CUDACC__) || defined(__CUDABE__) || defined( MOCK_CURAND ) || defined(MOCK_CUDA)
          
/**
qsim::propagate_at_surface   (HMM: perhaps propagate_at_simplified_surface )
-------------------------------------------------------------------------------

+---------------------+------------------+---------------------------------------------------------+--------------+
| output flag         |   command        |  changed                                                |  note        |
+=====================+==================+=========================================================+==============+
|   SURFACE_ABSORB    |    BREAK         |                                                         |              |
+---------------------+------------------+---------------------------------------------------------+--------------+
|   SURFACE_DETECT    |    BREAK         |                                                         |              |
+---------------------+------------------+---------------------------------------------------------+--------------+
|   SURFACE_DREFLECT  |    CONTINUE      |   direction, polarization                               |              |
+---------------------+------------------+---------------------------------------------------------+--------------+
|   SURFACE_SREFLECT  |    CONTINUE      |   direction, polarization                               |              |
+---------------------+------------------+---------------------------------------------------------+--------------+

Action depens on s.surface float4 params::

+---------------+---------------------+----------------------------------+
| s.surface.x   | detect fraction     |                                  |
+---------------+---------------------+----------------------------------+
| s.surface.y   | absorb fraction     |                                  |
+---------------+---------------------+----------------------------------+
| s.surface.z   | reflect specular    | NOT USED AS ALL FOUR SUM TO 1.   |               
+---------------+---------------------+----------------------------------+
| s.surface.w   | reflect diffuse     |                                  |
+---------------+---------------------+----------------------------------+

Excluding technical alignment burns a single u_surface random is throw
to decide on what happens at the surface according to the float4 probabilities 
that are assumed to sum to unity::
  
   +-------------------+-------------------+---------------------+------------------------+ 
   |                   |                   |                     |                        | 
   |  absorb           |  detect           |  reflect_diffuse_   |  "reflect_specular_"   |
   0--SURFACE_ABSORB---|--SURFACE_DETECT---|--SURFACE_DREFLECT---|--SURFACE_SREFLECT------1
   |  s.surface.y      |  s.surface.x      |  s.surface.w        |  s.surface.z           | 
   |                   |                   |                     |                        | 
   +-------------------+-------------------+---------------------+------------------------+ 

Refs::
 
    GSurfaceLib::propertyName
    GSurfaceLib::createStandardSurface

Different types of surface are modelled by changing the four values, 
in such a way that they always sum to unity::

    In [20]: t.oldsur.shape                                                                                          
    Out[20]: (46, 2, 761, 4)

    In [21]: t.oldsur[:,0,:].shape  ## pick payload group 0 
    Out[21]: (46, 761, 4)

    assert np.all( np.sum(t.oldsur[:,0,:], axis=2) == 1. ) 

The s.surface float4 is filled by qbnd::fill_state via::

    const int su_line = cosTheta > 0.f ? line + ISUR : line + OSUR ;
    s.surface = boundary_lookup(wavelength,su_line,0) ; 
    s.optical = optical[su_line].u ; 

**/

inline QSIM_METHOD int qsim::propagate_at_surface(unsigned& flag, curandStateXORWOW& rng, sctx& ctx)
{
    const sstate& s = ctx.s ; 
    const float& detect = s.surface.x ;
    const float& absorb = s.surface.y ;
    //const float& reflect_specular_ = s.surface.z ; 
    const float& reflect_diffuse_  = s.surface.w ; 

    float u_surface = curand_uniform(&rng);

#ifdef DEBUG_TAG
    stagr& tagr = ctx.tagr ; 
    float u_surface_burn = curand_uniform(&rng);
    tagr.add( stag_at_burn_sf_sd, u_surface); 
    tagr.add( stag_sf_burn,       u_surface_burn); 
#endif


    int action = u_surface < absorb + detect ? BREAK : CONTINUE  ; 

    if( action == BREAK )
    {
        flag = u_surface < absorb ? SURFACE_ABSORB : SURFACE_DETECT  ;
#ifdef DEBUG_PIDX
        if(ctx.idx == base->pidx)
        printf("//qsim.propagate_at_surface.SA/SD.BREAK idx %d : flag %d \n" , ctx.idx, flag ); 
#endif
    }
    else 
    {
        flag = u_surface < absorb + detect + reflect_diffuse_ ?  SURFACE_DREFLECT : SURFACE_SREFLECT ;  
        switch(flag)
        {
            case SURFACE_DREFLECT: reflect_diffuse( rng, ctx)  ; break ; 
            case SURFACE_SREFLECT: reflect_specular(rng, ctx)  ; break ;
        }
#ifdef DEBUG_PIDX
        if(ctx.idx == base->pidx)
        printf("//qsim.propagate_at_surface.DR/SR.CONTINUE idx %d : flag %d \n" , ctx.idx, flag ); 
#endif
    }
    return action ; 
}


inline QSIM_METHOD int qsim::propagate_at_surface_Detect(unsigned& flag, curandStateXORWOW& rng, sctx& ctx) const
{
    float u_surface_burn = curand_uniform(&rng);  // for random alignment 
    flag = SURFACE_DETECT ; 
    return BREAK ; 
}


#if defined(WITH_CUSTOM4)

/**
qsim::propagate_at_surface_CustomART
-------------------------------------

lpmtid:-1 
   indicates "not-a-sensor" and a sensor_identity issue, as CustomART 
   special surfaces are expected to always have sensor identities
 
**/

inline QSIM_METHOD int qsim::propagate_at_surface_CustomART(unsigned& flag, curandStateXORWOW& rng, sctx& ctx) const
{

    const sphoton& p = ctx.p ; 
    const float3* normal = (float3*)&ctx.prd->q0.f.x ;  // geometrical outwards normal 
    int lpmtid = ctx.prd->identity() - 1 ;  // identity comes from optixInstance.instanceId where 0 means not-a-sensor  
    float minus_cos_theta = dot(p.mom, *normal); 
    float dot_pol_cross_mom_nrm = dot(p.pol,cross(p.mom,*normal)) ; 

#ifdef DEBUG_PIDX
    if( ctx.idx == base->pidx ) 
    {
    float3 cross_mom_nrm = cross(p.mom, *normal) ; 
    printf("//qsim::propagate_at_surface_CustomART idx %7d : mom = np.array([%10.8f,%10.8f,%10.8f]) ; lmom = %10.8f \n", 
       ctx.idx, p.mom.x, p.mom.y, p.mom.z, length(p.mom) );  
    printf("//qsim::propagate_at_surface_CustomART idx %7d : pol = np.array([%10.8f,%10.8f,%10.8f]) ; lpol = %10.8f \n", 
       ctx.idx, p.pol.x, p.pol.y, p.pol.z, length(p.pol) );  
    printf("//qsim::propagate_at_surface_CustomART idx %7d : nrm = np.array([%10.8f,%10.8f,%10.8f]) ; lnrm = %10.8f \n", 
       ctx.idx, normal->x, normal->y, normal->z, length(*normal) );  
    printf("//qsim::propagate_at_surface_CustomART idx %7d : cross_mom_nrm = np.array([%10.8f,%10.8f,%10.8f]) ; lcross_mom_nrm = %10.8f  \n", 
           ctx.idx, cross_mom_nrm.x, cross_mom_nrm.y, cross_mom_nrm.z, length(cross_mom_nrm)  );  
    printf("//qsim::propagate_at_surface_CustomART idx %7d : dot_pol_cross_mom_nrm = %10.8f \n", ctx.idx, dot_pol_cross_mom_nrm ); 
    printf("//qsim::propagate_at_surface_CustomART idx %7d : minus_cos_theta = %10.8f \n", ctx.idx, minus_cos_theta ); 
    }
#endif

    if(lpmtid < 0 )
    {
        flag = NAN_ABORT ; 
#ifdef DEBUG_PIDX
        //if( ctx.idx == base->pidx ) 
        printf("//qsim::propagate_at_surface_CustomART idx %7d lpmtid %d : ERROR NOT-A-SENSOR : NAN_ABORT \n", ctx.idx, lpmtid ); 
#endif
        return BREAK ; 
    }


    float ARTE[4] ; 
    if(lpmtid > -1) pmt->get_lpmtid_ARTE(ARTE, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm );   

#ifdef DEBUG_PIDX
    if( ctx.idx == base->pidx ) 
    printf("//qsim::propagate_at_surface_CustomART idx %d lpmtid %d wl %7.3f mct %7.3f dpcmn %7.3f ARTE ( %7.3f %7.3f %7.3f %7.3f ) \n", 
           ctx.idx, lpmtid, p.wavelength, minus_cos_theta, dot_pol_cross_mom_nrm, ARTE[0], ARTE[1], ARTE[2], ARTE[3] );  
#endif


    const float& theAbsorption = ARTE[0]; 
    //const float& theReflectivity = ARTE[1]; 
    const float& theTransmittance = ARTE[2]; 
    const float& theEfficiency = ARTE[3]; 

    float u_theAbsorption = curand_uniform(&rng);
    int action = u_theAbsorption < theAbsorption  ? BREAK : CONTINUE ;


#ifdef DEBUG_PIDX
    if( ctx.idx == base->pidx ) 
    printf("//qsim.propagate_at_surface_CustomART idx %d lpmtid %d ARTE ( %7.3f %7.3f %7.3f %7.3f ) u_theAbsorption  %7.3f action %d \n", 
        ctx.idx, lpmtid, ARTE[0], ARTE[1], ARTE[2], ARTE[3], u_theAbsorption, action  );   
#endif
     
    if( action == BREAK )
    {
        float u_theEfficiency = curand_uniform(&rng) ; 
        flag = u_theEfficiency < theEfficiency ? SURFACE_DETECT : SURFACE_ABSORB ;  
    }
    else
    {
        propagate_at_boundary( flag, rng, ctx, theTransmittance  );  
    } 
    return action ; 
}
#endif

#endif


#if defined(__CUDACC__) || defined(__CUDABE__)  || defined(MOCK_CURAND) || defined(MOCK_CUDA) 

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




Orient the normal depending on incident momentum to handle inside/outside
--------------------------------------------------------------------------

HMM: Geant4 is very flippy with normals whereas Opticks uses fixed 
geometrical normals that may then need to be oriented for a specfic task, 
such as diffuse reflection from the shape. 


::

        geometrical normal
         : 
       \ : /  diffuse reflection from outside in same hemi as geometrical normal : orient 1.f 
        \:/   incident momentum is against the geometrical normal   dot( old_mom, *normal ) < 0.f 
       --*------+
      /         |
     /          |
     \          /
      \ \   /  /   diffuse reflection from inside needs to use a flipped geometrical normal : orient -1.f
       \ \ /  /    incident moment is with the geometrical normal   dot( old_mom, *normal ) > 0.f 
        \_*__/
          :
          :
          geometrical normal


orient is used to flip the reflection normal back against the incident direction

**/

inline QSIM_METHOD void qsim::reflect_diffuse( curandStateXORWOW& rng, sctx& ctx )
{
    sphoton& p = ctx.p ;  

    float3 old_mom = p.mom ; 

    const float3* normal = ctx.prd->normal() ;    // geometrical outwards normal 

    //const float orient = -1.f ; // BUG : FIXED ORIENT FLIP CANNOT BE CORRECT 
    const float orient = dot( old_mom, *normal ) > 0.f ? -1.f : 1.f ; 

    lambertian_direction( &p.mom, normal, orient, rng, ctx );


    float3 facet_normal = normalize( p.mom - old_mom ); 
    const float EdotN = dot( p.pol, facet_normal ); 
    p.pol = -1.f*(p.pol) + 2.f*EdotN*facet_normal ; 

#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.reflect_diffuse idx %d : old_mom = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  old_mom.x, old_mom.y, old_mom.z ) ; 

    printf("//qsim.reflect_diffuse idx %d : normal0 = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  normal->x, normal->y, normal->z ) ; 

    printf("//qsim.reflect_diffuse idx %d : p.mom = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  p.mom.x, p.mom.y, p.mom.z ) ; 

    printf("//qsim.reflect_diffuse idx %d : facet_normal = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  facet_normal.x, facet_normal.y, facet_normal.z ) ; 
    }
#endif

}

/**
qsim::reflect_specular
-----------------------

TODO: check the fixed orient flip with specular reflections
from inside and outside

Its not so obvious here because the oriented normal is part of the 
mom calc so it would be easy to double flip. 

G4OpBoundaryProcess::PostStepDoIt does an early flip of theGlobalNormal
but the G4Navigator does flips before that too... so Geant4 is too flippy 
to be helpful. 

**/

inline QSIM_METHOD void qsim::reflect_specular( curandStateXORWOW& rng, sctx& ctx )
{
    sphoton& p = ctx.p ;  
    const float3* normal = ctx.prd->normal() ;      

#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.reflect_specular.head idx %d : normal0 = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  normal->x, normal->y, normal->z ) ; 

    printf("//qsim.reflect_specular.head idx %d : mom0 = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  p.mom.x, p.mom.y, p.mom.z ) ; 

    printf("//qsim.reflect_specular.head idx %d : pol0 = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  p.pol.x, p.pol.y, p.pol.z ) ; 
    }
#endif

#ifdef WITH_ORIENT
    //const float orient = -1.f ;    
    const float orient = 1.f ;    
    // because orient appears twice in the below p.mom p.pol calcs 
    // it being +1.f or -1.f makes no difference

    const float PdotN = dot( p.mom, *normal )*orient ; 
    p.mom = p.mom - 2.f*PdotN*(*normal)*orient ;  

    const float EdotN = dot( p.pol, *normal )*orient ; 
    p.pol = -1.f*(p.pol) + 2.f*EdotN*(*normal)*orient  ; 
#else
    // removed orient as does not effect calc, hence confusing and pointless
    const float PdotN = dot( p.mom, *normal ) ; 
    p.mom = p.mom - 2.f*PdotN*(*normal) ;  

    const float EdotN = dot( p.pol, *normal ) ; 
    p.pol = -1.f*(p.pol) + 2.f*EdotN*(*normal)  ; 
#endif

#ifdef DEBUG_PIDX
    if(ctx.idx == base->pidx)
    {
    printf("//qsim.reflect_specular.tail idx %d : mom1 = np.array([%10.5f,%10.5f,%10.5f]) ; PdotN = %10.5f ; EdotN = %10.5f \n",
        ctx.idx,  p.mom.x, p.mom.y, p.mom.z, PdotN, EdotN  ) ; 

    printf("//qsim.reflect_specular.tail idx %d : pol1 = np.array([%10.5f,%10.5f,%10.5f]) \n",
        ctx.idx,  p.pol.x, p.pol.y, p.pol.z ) ; 

    }
#endif


}

/**
qsim::mock_propagate TODO: rename to mock_simulate 
------------------------------------------------------

This uses mock input prd (quad2) to provide a CUDA only (no OptiX, no geometry) 
test of qsim propagation machinery. 

* NB: qsim::mock_propagate is intended to be very similar to CSGOptiX/CSGOptiX7.cu::simulate 

TODO
~~~~~

* compare with cx/CSGOptiX7.cu::simulate and find users of this to see if it could be made more similar to cx::simulate 
* can sstate be slimmed : seems not very easily 
* simplify sstate persisting (quad6?quad5?)

Stages within bounce loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. mock call to OptiX trace : doing "geometry lookup"
   photon position + direction -> surface normal + distance and identity, boundary 

   * cosTheta sign gives boundary orientation   

2. lookup material/surface properties using boundary and orientation (cosTheta) from geometry lookup 

3. mutate photon and set flag using material properties

  * note that photons that SAIL to boundary are mutated twice within the while loop (by propagate_to_boundary and propagate_at_boundary/surface) 

Thoughts
~~~~~~~~~~~

NB: the different collection streams (photon, record, rec, seq) must stay independent 
so can switch them off easily in production running 


**/

inline QSIM_METHOD void qsim::mock_propagate( sphoton& p, const quad2* mock_prd, curandStateXORWOW& rng, unsigned idx )
{
    p.set_flag(TORCH);  // setting initial flag : in reality this should be done by generation

    qsim* sim = this ; 

    sctx ctx = {} ; 
    ctx.p = p ;     // Q: Why is this different from CSGOptiX7.cu:simulate ? A: Presumably due to input photon. 
    ctx.evt = evt ; 
    ctx.idx = idx ; 

    int command = START ; 
    int bounce = 0 ; 
#ifndef PRODUCTION
    ctx.point(bounce);  
#endif

    while( bounce < evt->max_bounce )
    {
        ctx.prd = mock_prd + (evt->max_bounce*idx+bounce) ;  
        if( ctx.prd->boundary() == 0xffffu ) break ;   // SHOULD NEVER HAPPEN : propagate can do nothing meaningful without a boundary 
#ifndef PRODUCTION
        ctx.trace(bounce);  
#endif

#ifdef DEBUG_PIDX
        if(idx == base->pidx) 
        printf("//qsim.mock_propagate idx %d bounce %d evt.max_bounce %d prd.q0.f.xyzw (%10.4f %10.4f %10.4f %10.4f) \n", 
             idx, bounce, evt->max_bounce, ctx.prd->q0.f.x, ctx.prd->q0.f.y, ctx.prd->q0.f.z, ctx.prd->q0.f.w );  
#endif
        command = sim->propagate(bounce, rng, ctx ); 
        bounce++;        
#ifndef PRODUCTION
        ctx.point(bounce); 
#endif
        if(command == BREAK) break ;    
    }
#ifndef PRODUCTION
    ctx.end();   
#endif
    evt->photon[idx] = ctx.p ;
}

/**
qsim::propagate : one "bounce" propagate_to_boundary/propagate_at_boundary 
-----------------------------------------------------------------------------

This is canonically invoked from within the bounce loop of CSGOptiX/OptiX7Test.cu:simulate 
after tracing (or mock tracing) has populated ctx.prd

MISSING intersects
~~~~~~~~~~~~~~~~~~~

Formerly thought "MISSING needs to return BREAK" but MISSING 
intersects only happen with "broken" test geometry so no need to be 
concerned with that. 

z-plus special sensor surfaces 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two kinds of z-plus sensor surfaces need to fallback to ordinary 
surface handling for z-minus, when lposcost < 0.f at lower z-hemi.  
Initially thought would need to use optical.z to refer to 
the fallback and call qbnd::fill_state again. 
But of course that is wrong, the osur/isur surface references 
already provide the z-minus fallback.
Hence can implement the fallback using standard surface branch
with no change to qbnd::fill_state. 
For z-plus with lposcost > 0.f upper z-hemi simply ignore 
the s.surface

traditional POM special surface : smatsur_Surface_zplus_sensor_A
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HMM: could use standard surface handling for this but with perfect
detector surface swapped in.  


TMM multilayer POM special surface : smatsur_Surface_zplus_sensor_CustomART
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enabling special surface handling is controlled by ctx.optical.y "ems" enum 
which wheels in the qpmt.h stack calc following C4CustomART::doIt 
for CustomART special surfaces. 

Prior to supporting special surfaces, within the command == BOUNDARY used::

    command = ctx.s.optical.x == 0 ? 
                                      propagate_at_boundary( flag, rng, ctx ) 
                                  : 
                                      propagate_at_surface( flag, rng, ctx ) 
                                  ;  

**/

inline QSIM_METHOD int qsim::propagate(const int bounce, curandStateXORWOW& rng, sctx& ctx ) 
{
    const unsigned boundary = ctx.prd->boundary() ; 
    const unsigned identity = ctx.prd->identity() ; 
    const unsigned iindex = ctx.prd->iindex() ; 
    const float lposcost = ctx.prd->lposcost() ;  // local frame intersect position cosine theta 

    const float3* normal = ctx.prd->normal(); 
    float cosTheta = dot(ctx.p.mom, *normal ) ;    

#ifdef DEBUG_PIDX
    if( ctx.idx == base->pidx ) 
    {
    printf("\n//qsim.propagate.head idx %d : bnc %d cosTheta %10.8f \n", ctx.idx, bounce, cosTheta ); 

    printf("//qsim.propagate.head idx %d : mom = np.array([%10.8f,%10.8f,%10.8f]) ; lmom = %10.8f  \n", 
                 ctx.idx, ctx.p.mom.x, ctx.p.mom.y, ctx.p.mom.z, length(ctx.p.mom) ) ; 

    printf("//qsim.propagate.head idx %d : pos = np.array([%10.5f,%10.5f,%10.5f]) ; lpos = %10.8f \n", 
                 ctx.idx, ctx.p.pos.x, ctx.p.pos.y, ctx.p.pos.z, length(ctx.p.pos) ) ; 

    printf("//qsim.propagate.head idx %d : nrm = np.array([(%10.8f,%10.8f,%10.8f]) ; lnrm = %10.8f  \n", 
                 ctx.idx, normal->x, normal->y, normal->z, length(*normal) ); 

    }
#endif

    ctx.p.set_prd(boundary, identity, cosTheta, iindex );  // HMM: lposcost not passed along 

    bnd->fill_state(ctx.s, boundary, ctx.p.wavelength, cosTheta, ctx.idx ); 

    unsigned flag = 0 ;  

    int command = propagate_to_boundary( flag, rng, ctx );  
    /**
    command possibilities:

    1. CONTINUE (eg scatter)
    2. BREAK (eg absorb) 
    3. BOUNDARY 

    **/

#ifdef DEBUG_PIDX
    if( ctx.idx == base->pidx ) 
    printf("//qsim.propagate idx %d bounce %d command %d flag %d s.optical.x %d s.optical.y %d \n", 
          ctx.idx, bounce, command, flag, ctx.s.optical.x, ctx.s.optical.y );   
#endif

    if( command == BOUNDARY )
    {
        const int& ems = ctx.s.optical.y ; 

#if defined(DEBUG_PIDX) 
        if( ctx.idx == base->pidx ) 
        {
#if defined(WITH_CUSTOM4)
            printf("//qsim.propagate.WITH_CUSTOM4 idx %d  BOUNDARY ems %d lposcost %7.3f \n", ctx.idx, ems, lposcost ); 
#else
            printf("//qsim.propagate.NOT:WITH_CUSTOM4 idx %d BOUNDARY ems %d %lposcost %7.3f \n", ctx.idx, ems, lposcost); 
#endif
        }
#endif

        if( ems == smatsur_NoSurface )
        {
            command = propagate_at_boundary( flag, rng, ctx ) ; 
        }
        else if( ems == smatsur_Surface )
        {
            command = propagate_at_surface( flag, rng, ctx ) ; 
        }
        else if( lposcost < 0.f )  // could combine with prior, but handy for debug to keep separate
        {
#ifdef DEBUG_PIDX
            if( ctx.idx == base->pidx ) 
            printf("//qsim.propagate (lposcost < 0.f) idx %d bounce %d command %d flag %d ems %d \n", 
                     ctx.idx, bounce, command, flag, ems  );   
#endif           
            command = propagate_at_surface( flag, rng, ctx ) ; 
        }
        else if( ems == smatsur_Surface_zplus_sensor_A )
        {
            command = propagate_at_surface_Detect( flag, rng, ctx ) ; 
        }
        else if( ems == smatsur_Surface_zplus_sensor_CustomART )
        {
#if defined(WITH_CUSTOM4)
            command = propagate_at_surface_CustomART( flag, rng, ctx ) ; 
#endif
        }
    }
    ctx.p.set_flag(flag);   
    return command ; 
}
/**
Q: Where does ctx.s.optical come from ?
A: Populated in qbnd::fill_state based on boundary and cosTheta sign to get the 
   su_line index of the optical buffer which distinguishes surface from boundary,
   as acted upon above. See also GBndLib::createOpticalBuffer   

Q: Can the big "if" above be changed into a switch ?
A: YES, but non-trivially and probably confusingly. This is because 
   the special surface handling condition needs to be after the possibiliy 
   of fallback to ordinary surface handling for "lposcost < 0.f" and also 
   ordinary NoSurface must be possible no matter the lposcost value. 

**/



/**
qsim::hemisphere_polarized
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

inline QSIM_METHOD void qsim::hemisphere_polarized( unsigned polz, bool inwards, curandStateXORWOW& rng, sctx& ctx )
{
    sphoton& p = ctx.p ; 
    const float3* normal = ctx.prd->normal() ; 

    //printf("//qsim.hemisphere_polarized polz %d normal (%10.4f, %10.4f, %10.4f) \n", polz, normal->x, normal->y, normal->z );  

    float u_hemipol_phi = curand_uniform(&rng) ; 
    float phi = u_hemipol_phi*2.f*M_PIf;  // 0->2pi
    float cosTheta = curand_uniform(&rng) ;      // 0->1

#ifdef DEBUG_TAG
    stagr& tagr = ctx.tagr ; 
    tagr.add( stag_hp_ph, u_hemipol_phi ); 
    tagr.add( stag_hp_ph, cosTheta );    // trying to reduce stag::BITS from 5 to 4, so change stag_hp_ct to stag_hp_ph 
#endif

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

* NB the sxyz.h enum order is different to the python one  eg XYZ=0 
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

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    __sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);
#else
    sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);
#endif

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

Moved non-standard center-extent (aka frame) gensteps to use qsim::generate_photon_simtrace not this 

**/

inline QSIM_METHOD void qsim::generate_photon(sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const 
{
    const int& gencode = gs.q0.i.x ; 
    switch(gencode)
    {
        case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ; 
        case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ; 

        case OpticksGenstep_G4Cerenkov_modified:
        case OpticksGenstep_CERENKOV:        
                                              cerenkov->generate(    p, rng, gs, photon_id, genstep_id ) ; break ; 

        case OpticksGenstep_DsG4Scintillation_r4695:
        case OpticksGenstep_SCINTILLATION:   
                                              scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ; 

        case OpticksGenstep_INPUT_PHOTON:    { p = evt->photon[photon_id] ; p.set_flag(TORCH) ; }        ; break ;        
        default:                             generate_photon_dummy(  p, rng, gs, photon_id, genstep_id)  ; break ; 
    }
}
#endif

