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

Hmm:

* qctx encompasses global info relevant to to all photons, making any changes
  to it from single threads must only be into thread-owned slots to avoid interference 
 
* temporary working state local to each photon is currently being passed by reference args, 
  would be cleaner to use a collective state struct to hold this local structs 

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

    QCTX_METHOD float4  boundary_lookup( unsigned ix, unsigned iy ); 

    QCTX_METHOD float   scint_wavelength_hd0(curandStateXORWOW& rng);  
    QCTX_METHOD float   scint_wavelength_hd10(curandStateXORWOW& rng);
    QCTX_METHOD float   scint_wavelength_hd20(curandStateXORWOW& rng);
    QCTX_METHOD void    scint_dirpol(quad4& p, curandStateXORWOW& rng); 
    QCTX_METHOD void    reemit_photon(quad4& p, float scintillationTime, curandStateXORWOW& rng);
    QCTX_METHOD void    scint_photon( quad4& p, GS& g, curandStateXORWOW& rng);

    QCTX_METHOD float   cerenkov_wavelength(const GS& g, curandStateXORWOW& rng);
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

inline QCTX_METHOD float4 qctx::boundary_lookup( unsigned ix, unsigned iy )
{
    unsigned nx = boundary_meta->q0.u.x  ; 
    unsigned ny = boundary_meta->q0.u.y  ; 
    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;
    float4 props = tex2D<float4>( boundary_tex, x, y );     
    return props ; 
}

inline QCTX_METHOD float qctx::scint_wavelength_hd0(curandStateXORWOW& rng) 
{
    constexpr float y0 = 0.5f/3.f ; 
    float u0 = curand_uniform(&rng); 
    return tex2D<float>(scint_tex, u0, y0 );    
}

/**
qctx::scint_wavelength_hd10
--------------------------------------------------

Idea is to improve handling of extremes by throwing ten times the bins
at those regions, using simple and cheap linear mappings.

Perhaps could also use log probabilities to do something similar to 
this in a fancy way : just like using log scale to give more detail in the low registers. 
But that has computational disadvantage of expensive mapping functions to get between spaces. 

See::

     extg4/tests/X4ScintillatorTest.cc
     extg4/tests/X4ScintillatorTest.py 


Actually looking at the plots would be better to do the extreme splits at 0.05 and 0.95 

**/

inline QCTX_METHOD float qctx::scint_wavelength_hd10(curandStateXORWOW& rng) 
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



inline QCTX_METHOD float qctx::scint_wavelength_hd20(curandStateXORWOW& rng) 
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

inline QCTX_METHOD float qctx::cerenkov_wavelength(const GS& g, curandStateXORWOW& rng) 
{
    float u ; 
    float w ; 
    float wavelength ;
    float sampledRI ;
    float cosTheta ;
    float sin2Theta ;
    float u_maxSin2 ;

    do {

        u = curand_uniform(&rng) ;

        w = g.ck1.Wmin + u*(g.ck1.Wmax - g.ck1.Wmin) ; // avoid lerp compilation issue

        wavelength = g.ck1.Wmin*g.ck1.Wmax/w ;  

        // wavelength between Wmin and Wmax uniform-reciprocal-sampled to mimic uniform energy range sampling 

        float4 props = boundary_lookup(0, 0); 
        //float4 props = boundary_lookup(wavelength, g.st.MaterialIndex, 0); 

        sampledRI = props.x ;

        cosTheta = g.ck1.BetaInverse / sampledRI ;

        sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  // avoid going -ve 

        u = curand_uniform(&rng) ;

        u_maxSin2 = u*g.ck1.maxSin2 ;

    } while ( u_maxSin2 > sin2Theta);

    return wavelength ; 
}




/**
qctx::scint_dirpol
--------------------

Fills the photon quad4 struct with the below:

* direction, weight
* polarization, wavelength 

NB no position, time.

**/

inline QCTX_METHOD void qctx::scint_dirpol(quad4& p, curandStateXORWOW& rng)
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


epsilon:podio blyth$ jsc
2 files to edit
./Simulation/DetSimV2/PhysiSim/include/DsG4Scintillation.h
./Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc
epsilon:offline blyth$ jsc
2 files to edit
./Simulation/DetSimV2/PhysiSim/include/DsG4Scintillation.h
./Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc
epsilon:offline blyth$ jgr OpticalCONSTANT
./Simulation/DetSimV2/PhysiSim/src/DsG4ScintSimple.cc:      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
./Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc:      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        LSMPT->AddProperty("OpticalCONSTANT",OpticalTimeConstant,OpticalYieldRatio,1);
./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        helper_mpt(LSMPT, "OpticalCONSTANT",         mcgt.data(), "Material.LS.OpticalCONSTANT");
epsilon:offline blyth$ jgr OpticalTimeConstant
./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        LSMPT->AddProperty("OpticalCONSTANT",OpticalTimeConstant,OpticalYieldRatio,1);
./Simulation/DetSimV2/DetSimOptions/src/OpticalProperty.icc:  double OpticalTimeConstant[1] = {1.50*ns };
epsilon:offline blyth$ 

**/

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

