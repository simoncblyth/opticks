#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QCTX_METHOD __device__
#else
   #define QCTX_METHOD 
#endif 

#include "qgs.h"

/**
qctx
=====

This is aiming to replace the OptiX 6 context in a CUDA-centric way.

**/

struct curandStateXORWOW ; 

struct qctx
{
    curandStateXORWOW*  r ; 

    cudaTextureObject_t scint_tex ; 
    quad4*              scint_meta ;

    cudaTextureObject_t boundary_tex ; 
    quad4*              boundary_meta ; 

    quad6*              genstep ; 
    unsigned            genstep_id ; 

    quad4*              photon ; 
    unsigned            photon_id ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QCTX_METHOD float   scint_wavelength(curandStateXORWOW& rng);  
    QCTX_METHOD void    scint_dirpol(quad4& p, curandStateXORWOW& rng); 
    QCTX_METHOD void    reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng);
    QCTX_METHOD void    scint_photon( quad4& p, GS& g, curandStateXORWOW& rng);
#else
    qctx()
        :
        r(nullptr),
        scint_tex(0),
        scint_meta(nullptr),
        boundary_tex(0),
        boundary_meta(nullptr),
        genstep(nullptr),
        genstep_id(~0u),
        photon(nullptr),
        photon_id(~0u)
    {
    }
#endif

}; 


// TODO: get the below to work on CPU with mocked curand and tex2D

#if defined(__CUDACC__) || defined(__CUDABE__)
inline QCTX_METHOD float qctx::scint_wavelength(curandStateXORWOW& rng) 
{
    float u0 = curand_uniform(&rng); 
    return tex2D<float>(scint_tex, u0, 0.f);    
}

inline QCTX_METHOD void qctx::scint_dirpol(quad4& p, curandStateXORWOW& rng)
{
    float u0 = curand_uniform(&rng) ; 
    float u1 = curand_uniform(&rng) ; 
    float u2 = curand_uniform(&rng) ;   
    float u3 = curand_uniform(&rng) ;   

    float wavelength = tex2D<float>(scint_tex, u0, 0.f);
    float weight = 1.f ; 

    float ct = 1.0f - 2.0f*u1 ;                 // -1.: 1. 
    float st = sqrtf( (1.0f-ct)*(1.0f+ct)) ; 
    float phi = 2.f*M_PIf*u2 ;

    float sp = sinf(phi); 
    float cp = cosf(phi); 

    float3 dir0 = make_float3( st*cp, st*sp,  ct ); 

    p.q1.f.x = dir0.x ; 
    p.q1.f.y = dir0.y ; 
    p.q1.f.z = dir0.z ; 
    p.q1.f.w = weight ;  

    float3 pol0 = make_float3( ct*cp, ct*sp, -st );
    float3 perp = cross( dir0, pol0 ); 

    float az =  2.f*M_PIf*u3 ; 
    float sz = sin(az);
    float cz = cos(az);

    float3 pol1 = normalize( cz*pol0 + sz*perp ) ; 

    p.q2.f.x = pol1.x ; 
    p.q2.f.y = pol1.y ; 
    p.q2.f.z = pol1.z ; 
    p.q2.f.w = wavelength ; 
}

inline QCTX_METHOD void qctx::reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng)
{
    scint_dirpol(p, rng); 
    float u4 = curand_uniform(&rng) ; 
    p.q0.f.w += -scintillationTime*logf(u4) ;
}

inline QCTX_METHOD void qctx::scint_photon(quad4& p, GS& g, curandStateXORWOW& rng)
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
#endif

