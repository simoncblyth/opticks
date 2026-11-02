#pragma once
/**
qscint_three.h
===============

This is not named "qscintthree.h" as that is only diffent by case to "QScintThree.h"

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSCINT_METHOD __device__
#else
   #define QSCINT_METHOD
#endif


#include "qrng.h"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "sscint.h"
#include "sspecies.h"
#include "OpticksPhoton.h"


struct qscint_three
{
    cudaTextureObject_t icdf ;
    cudaTextureObject_t abslength ;
    cudaTextureObject_t reemissionprob ;

    // domain of the abslength/reemissionprob textures (from prop_three metas, U4ScintThreeExtra::MakePropArray)
    float    nm0 ;                    // domain_low  [nm]
    float    nms ;                    // domain_step [nm]
    unsigned nx  ;                    // number of wavelength samples

    // 0-based index of LS within the materials array:
    // kernel is_ls = ( s.index.x == ls_material_index + 1u )
    unsigned ls_material_index ;

    // LS new-optical-model constants indexed by species (from the const_three array)
    float m_tau_intrinsic[2] ;    // PPO_INTRINSICTAU, BISMSB_INTRINSICTAU
    float m_prob_lab2[2] ;        // LAB2PPO_PROB    , LAB2BIS_PROB
    float m_tau_lab2[2] ;         // LAB2PPO_TAU     , LAB2BIS_TAU

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
    QSCINT_METHOD float   wavelength_sd(int species_idx_, float u0) const ;
    QSCINT_METHOD float   wavelength_hd20_without_margin(int species_idx_, float u0) const ;


    QSCINT_METHOD static  float BinCentered(float x, float N) ;
    QSCINT_METHOD float   wavelength_hd20(int species_idx_, float u0) const ;

    QSCINT_METHOD float4  abslength_lookup(float nm) const ;
    QSCINT_METHOD float4  reemissionprob_lookup(float nm) const ;

    QSCINT_METHOD static  int SpeciesIndex_(int scnt) ;
    // scnt 0/1 -> 0 (PPO), 2/3 -> 1 (BISMSB)

    QSCINT_METHOD static  void sample_direction_polarization(float3& dir, float3& pol, RNG& rng) ;
    // translation of CPU DsOpticalModelHelper::SampleDirectionAndPolarization

    QSCINT_METHOD         void generate( sphoton& p, RNG& rng, const quad6& gs, unsigned long long photon_id, int genstep_id ) const ;
    // GPU photon generation for OpticksGenstep_DsG4ScintSimple. Mirrors the CPU
    // Local_DsG4ScintSimple photon loop step-for-step
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
inline QSCINT_METHOD float qscint_three::wavelength_sd(int species_idx_, float u0) const
{
    // no longer valid following zone change LHS//MID//RHS
    int layer = (species_idx_ * 3) ;
    return tex2DLayered<float>(icdf, u0, 0.5f, layer);
}

/**
qscint_three::wavelength_hd20_without_margin
---------------------------------------------

NB this _hd20 method is tied to the form and layout of the texture icdf
and its input array prepared by U4ScintCommon::CreateGeant4InterpolatedInverseCDF
as instructed from U4ScintThree::U4ScintThree.
Due to this it does not make sense to have a _hd10 method too, as would need a
separate texture to support that.
Because the _sd method simply uses the full range layer there is no problem with
that using the full 6 layer texture atlas and just ignoring the zoomed HD layers.

**/

inline QSCINT_METHOD float qscint_three::wavelength_hd20_without_margin(int species_idx_, float u0) const
{
    // no longer valid following zone change LHS//MID//RHS

    // 1. Determine HD region
    const bool is_lhs = (u0 < 0.05f);
    const bool is_rhs = (u0 > 0.95f);
    int hd_region_idx = is_lhs ? 1 : (is_rhs ? 2 : 0);

    // 2. Map u0 to the zoom coordinate x
    const float x = is_lhs ? (u0 * 20.f) : (is_rhs ? (u0 - 0.95f) * 20.f : u0);

    // 3. Calculate the absolute layer in the 6-layer atlas
    int layer = (species_idx_ * 3) + hd_region_idx;

    // 4. One-shot hardware fetch
    return tex2DLayered<float>(icdf, x, 0.5f, layer);
}

/**
qscint_three::BinCentered
--------------------------


**/



/**
qscint_three::wavelength_hd20
-------------------------------

NB this kernel code is tightly coupled with the code that prepares the
multi-resolution texture atlas in u4/U4ScintCommon.h
U4ScintCommon::CreateGeant4InterpolatedInverseCDF





HD20  e = 0.05
     2e = 0.1
      m = 0.005
     2m = 0.01



         0               e-m   e   e+m                                  1-e-m 1-e 1-e+m                    1
         |                |    |    |                                     |    |    |                      |

         +-------LHS----------------+                                     +----------RHS-------------------+
                          +---------------------MID---------------------------------+
                          |                                                         |
                          |                                                         |
                          |   (1-e+m) - (e-m)  = 1 - 2*e + 2m  =  1 - 2*(e-m)       |
                          |   edges trimmed with margins added                      |
                          |                                                         |
                          |                                                         |



    u0 < e-m : LHS-only region
       x = u0 / (e+m )     ## NB using actual range of the LHS texture for the mapping

    u0 [e-m -> e+m] : LHS+MID overlap region












**/


#ifdef WITH_LERP
inline QSCINT_METHOD float qscint_three::wavelength_hd20(int species_idx_, float u0_) const
{
    float u0 = fminf(0.999999f, fmaxf(0.000000f, u0_));

    const float e = 0.05f;
    const float m = 0.005f;
    const float N = 4096.f ;

    // 1. Pre-calculate Main mapping since it is used in multiple blend cases
    // Mapping u0 range [e-m, 1-e+m] to [0, 1]
    const float mid_denom = 1.0f - 2.0f * (e - m);
    const float x_mid = (u0 - (e - m)) / mid_denom;

    // 2. Branching logic with consolidated sampling
    if (u0 < e + m)
    {
        float x_lhs = u0 / (e + m);
        float v_lhs = tex2DLayered<float>(icdf, BinCentered(x_lhs,N), 0.5f, (species_idx_ * 3) + 1);
        if (u0 < e - m) return v_lhs;    // pure LHS

        float v_mid = tex2DLayered<float>(icdf, BinCentered(x_mid,N), 0.5f, (species_idx_ * 3) + 0);
        float t = (u0 - (e - m)) / (2.0f * m);  // fraction into LHS/MID overlap band
        return lerp(v_lhs, v_mid, t);           // Blend LHS+MID
    }
    else if (u0 > (1.0f - e - m))
    {
        float x_rhs = (u0 - (1.0f - e - m)) / (e + m);
        float v_rhs = tex2DLayered<float>(icdf, BinCentered(x_rhs,N), 0.5f, (species_idx_ * 3) + 2);
        if (u0 > (1.0f - e + m)) return v_rhs;  // pure RHS

        float v_mid = tex2DLayered<float>(icdf, BinCentered(x_mid,N), 0.5f, (species_idx_ * 3) + 0);
        float t = (u0 - (1.0f - e - m)) / (2.0f * m);   // fraction into MID/RHS overlap band
        return lerp(v_mid, v_rhs, t);                   // Blend MID+RHS
    }
    // Pure MID Zone
    return tex2DLayered<float>(icdf, BinCentered(x_mid,N), 0.5f, (species_idx_ * 3) + 0);
}
#else


/**
qscint_three::wavelength_hd20
-------------------------------

HD20  edge = 0.05   margin = 0.005    4096 bins


         0               e-m   e   e+m                                  1-e-m 1-e 1-e+m                    1
         |                |    |    |                                     |    |    |                      |

         +-------LHS----------------+                                     +----------RHS-------------------+
                          +---------------------MID---------------------------------+

                          |   (1-e+m) - (e-m)  = 1 - 2*e + 2m  =  1 - 2*(e-m)       |


    u0 < e  : LHS zone
       x = u0 / (e+m )

    u0 > 1.f - e : RHS zone
       x = ( u0 - (1.f - e - m)) / (e+m)

    otherwise : MID zone
       x = (u0 - (e - m)) / ( 1.0f - 2.0f * (e - m) );

**/



inline QSCINT_METHOD float qscint_three::wavelength_hd20(int species_idx_, float u0_) const
{
    float u0 = fminf(0.999999f, fmaxf(0.000000f, u0_));

    const float e = 0.05f;
    const float m = 0.005f;
    const float N = 4096.f ;

    float x ;
    int zone_idx ;

    if( u0 < e ) // LHS
    {
        x = u0 / (e + m);
        zone_idx = 1;
    }
    else if( u0 > 1.f - e)  // RHS
    {
        x = (u0 - (1.0f - e - m)) / (e + m);
        zone_idx = 2;
    }
    else  // MID
    {
        x = (u0 - (e - m)) / ( 1.0f - 2.0f * (e - m) );
        zone_idx = 0;
    }

    return tex2DLayered<float>(icdf, BinCentered(x,N), 0.5f, (species_idx_ * 3) + zone_idx );
}
#endif


/**
BinCentered

              0.5           N - 1
   x_tex =   ------ + u0 * -------
               N              N



                      0.5
    x_tex(u0=0) =    -----
                       N


                         0.5 + N  - 1           N - 0.5
    x_tex(u0=1)  =     --------------- =   ------------------
                              N                    N


    x = fmaxf(x , 0.5f / N) ;              // prevent going below 1st texel
    x = fminf( x, ( N - 0.5f )/N ) ;       // prevent going above last texel


**/


inline QSCINT_METHOD float qscint_three::BinCentered(float x, float N)  // static
{
    x = fmaxf(0.f, fminf(1.f, x)); // clamp logical x to [0,1] domain
    return (x*(N - 1.f) + 0.5f)/N ; // map logical [0,1] to texel coordinate [0.5/N,(N-0.5)/N]
    // this arranges for logical x=0(1) to hit center of texel 0(N-1)
}


/**
qscint_three::abslength_lookup / reemissionprob_lookup
-------------------------------------------------------

Lookups of the two rows of the LS-only prop_three texture data,
addressing mirrors qbnd::boundary_lookup::

    fx = (nm - nm0)/nms
    x  = (fx + 0.5f)/float(nx)

The texture height is 1, so the y-coordinate is always 0.5f.

Return float4::

    abslength_lookup      : ( fraction_lab, fraction_ppo, fraction_bis, abslength_combined )
    reemissionprob_lookup : ( reemission_prob_ppo, reemission_prob_bis, 0., 0. )

**/

inline QSCINT_METHOD float4 qscint_three::abslength_lookup(float nm) const
{
    float x = ((nm - nm0)/nms + 0.5f)/float(nx) ;
    return tex2D<float4>( abslength, x, 0.5f );
}

inline QSCINT_METHOD float4 qscint_three::reemissionprob_lookup(float nm) const
{
    float x = ((nm - nm0)/nms + 0.5f)/float(nx) ;
    return tex2D<float4>( reemissionprob, x, 0.5f );
}


inline QSCINT_METHOD int qscint_three::SpeciesIndex_(int scnt)  // static
{
    int species_idx_ = scnt < 2 ? 0 : 1 ;    // 0,1 -> 0 (PPO), 2,3 -> 1 (BISMSB)
    return species_idx_ ;
}


/**
qscint_three::sample_direction_polarization
--------------------------------------------

Translation of CPU DsOpticalModelHelper::SampleDirectionAndPolarization
(u4/Local_DsOpticalModelHelper.hh):

**/

inline QSCINT_METHOD void qscint_three::sample_direction_polarization(float3& dir, float3& pol, RNG& rng)  // static
{
    float u, v, S ;
    do {
        u = 2.f*curand_uniform(&rng) - 1.f ;
        v = 2.f*curand_uniform(&rng) - 1.f ;
        S = u*u + v*v ;
    } while ( S >= 1.f || S == 0.f ) ;

    float fac = 2.f*sqrtf(1.f - S) ;
    dir.x = u*fac ;
    dir.y = v*fac ;
    dir.z = 1.f - 2.f*S ;

    float ax = fabsf(dir.x);
    float ay = fabsf(dir.y);
    float az = fabsf(dir.z);
    float3 perp ;
    if( ax <= ay && ax <= az )
    {
        float norm = sqrtf(1.f - dir.x*dir.x);
        perp.x = 0.f;         perp.y = dir.z/norm; perp.z = -dir.y/norm;
    }
    else if( ay <= ax && ay <= az )
    {
        float norm = sqrtf(1.f - dir.y*dir.y);
        perp.x = -dir.z/norm; perp.y = 0.f;        perp.z = dir.x/norm;
    }
    else
    {
        float norm = sqrtf(1.f - dir.z*dir.z);
        perp.x = -dir.y/norm; perp.y = dir.x/norm; perp.z = 0.f;
    }

    float3 perp2 = cross(dir, perp);

    float phi = 2.f*M_PIf*curand_uniform(&rng);
    float cosp = cosf(phi);
    float sinp = sinf(phi);

    pol = cosp*perp + sinp*perp2 ;
}


/**
qscint_three::generate
-----------------------

GPU photon generation for a OpticksGenstep_DsG4ScintSimple genstep.
Mirrors the CPU Local_DsG4ScintSimple PostStepDoIt photon loop
(u4/Local_DsG4ScintSimple.cc) step for step.

**/

inline QSCINT_METHOD void qscint_three::generate(
    sphoton& p,
    RNG& rng,
    const quad6& _gs,
    unsigned long long photon_id,
    int genstep_id ) const
{
    const sscint& gs = (const sscint&)_gs ;
    int species_idx_ = SpeciesIndex_(gs.scnt) ;                  // 0:PPO, 1:BISMSB (0-based)
    unsigned species_idx = species_idx_ == 0 ? PPO : BISMSB ;    // photon tag (sspecies.h)

    // 1. wavelength
    float u_wl = curand_uniform(&rng) ;
    p.wavelength = wavelength_hd20(species_idx_, u_wl);

    // 2. isotropic direction + polarization
    sample_direction_polarization(p.mom, p.pol, rng);

    // 3. time: flight time along step + NR exponential delay only when tau>0
    // 4. position along the step
    float rand = gs.charge == 0.f ? 1.f : curand_uniform(&rng) ;

    float deltaTime = rand*gs.step_length/gs.meanVelocity ;

    if( gs.ScintillationTime > 0.f )
    {
        deltaTime += -logf(curand_uniform(&rng))*gs.ScintillationTime ;
    }

    p.pos = gs.pos + rand*gs.DeltaPosition ;

    p.time = gs.time + deltaTime ;
    p.zero_flags();
    p.set_flag(SCINTILLATION) ;
    p.set_species(species_idx) ;
}


#endif

