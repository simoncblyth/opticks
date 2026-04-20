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


struct qscint_three
{
    cudaTextureObject_t scint_tex_layered ;

#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
    QSCINT_METHOD float   wavelength_sd(int species_idx, float u0) const ;
    QSCINT_METHOD float   wavelength_hd20_without_margin(int species_idx, float u0) const ;


    QSCINT_METHOD static  float BinCentered(float x, float N) ;
    QSCINT_METHOD float   wavelength_hd20(int species_idx, float u0) const ;
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
inline QSCINT_METHOD float qscint_three::wavelength_sd(int species_idx, float u0) const
{
    // no longer valid following zone change LHS//MID//RHS
    int layer = (species_idx * 3) ;
    return tex2DLayered<float>(scint_tex_layered, u0, 0.5f, layer);
}

/**
qscint_three::wavelength_hd20_without_margin
---------------------------------------------

NB this _hd20 method is tied to the form and layout of the texture scint_tex_layered
and its input array prepared by U4ScintCommon::CreateGeant4InterpolatedInverseCDF
as instructed from U4ScintThree::U4ScintThree.
Due to this it does not make sense to have a _hd10 method too, as would need a
separate texture to support that.
Because the _sd method simply uses the full range layer there is no problem with
that using the full 9 layer texture atlas and just ignoring the zoomed HD layers.

**/

inline QSCINT_METHOD float qscint_three::wavelength_hd20_without_margin(int species_idx, float u0) const
{
    // no longer valid following zone change LHS//MID//RHS

    // 1. Determine HD region
    const bool is_lhs = (u0 < 0.05f);
    const bool is_rhs = (u0 > 0.95f);
    int hd_region_idx = is_lhs ? 1 : (is_rhs ? 2 : 0);

    // 2. Map u0 to the zoom coordinate x
    const float x = is_lhs ? (u0 * 20.f) : (is_rhs ? (u0 - 0.95f) * 20.f : u0);

    // 3. Calculate the absolute layer in the 9-layer atlas
    int layer = (species_idx * 3) + hd_region_idx;

    // 4. One-shot hardware fetch
    return tex2DLayered<float>(scint_tex_layered, x, 0.5f, layer);
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
inline QSCINT_METHOD float qscint_three::wavelength_hd20(int species_idx, float u0_) const
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
        float v_lhs = tex2DLayered<float>(scint_tex_layered, BinCentered(x_lhs,N), 0.5f, (species_idx * 3) + 1);
        if (u0 < e - m) return v_lhs;    // pure LHS

        float v_mid = tex2DLayered<float>(scint_tex_layered, BinCentered(x_mid,N), 0.5f, (species_idx * 3) + 0);
        float t = (u0 - (e - m)) / (2.0f * m);  // fraction into LHS/MID overlap band
        return lerp(v_lhs, v_mid, t);           // Blend LHS+MID
    }
    else if (u0 > (1.0f - e - m))
    {
        float x_rhs = (u0 - (1.0f - e - m)) / (e + m);
        float v_rhs = tex2DLayered<float>(scint_tex_layered, BinCentered(x_rhs,N), 0.5f, (species_idx * 3) + 2);
        if (u0 > (1.0f - e + m)) return v_rhs;  // pure RHS

        float v_mid = tex2DLayered<float>(scint_tex_layered, BinCentered(x_mid,N), 0.5f, (species_idx * 3) + 0);
        float t = (u0 - (1.0f - e - m)) / (2.0f * m);   // fraction into MID/RHS overlap band
        return lerp(v_mid, v_rhs, t);                   // Blend MID+RHS
    }

    // Pure MID Zone
    return tex2DLayered<float>(scint_tex_layered, BinCentered(x_mid,N), 0.5f, (species_idx * 3) + 0);
}
#else

inline QSCINT_METHOD float qscint_three::wavelength_hd20(int species_idx, float u0_) const
{
    float u0 = fminf(0.999999f, fmaxf(0.000000f, u0_));

    const float e = 0.05f;
    const float m = 0.005f;
    const float N = 4096.f ;

    bool is_lhs = u0 < e ;
    bool is_rhs = u0 > (1.f - e) ;

    float x_lhs = u0 / (e + m);
    float x_rhs = (u0 - (1.0f - e - m)) / (e + m);
    float x_mid = (u0 - (e - m)) / ( 1.0f - 2.0f * (e - m) );

    float x = is_lhs ? x_lhs : (  is_rhs ? x_rhs : x_mid ) ;
    int zone_idx = is_lhs ? 1 : ( is_rhs ? 2 : 0 )  ;

    return tex2DLayered<float>(scint_tex_layered, BinCentered(x,N), 0.5f, (species_idx * 3) + zone_idx );
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




**/


inline QSCINT_METHOD float qscint_three::BinCentered(float x, float N)  // static
{
    x = fmaxf(x , 0.5f / N) ;              // prevent going below 1st texel
    x = fminf( x, ( N - 0.5f )/N ) ;       // prevent going above last texel
    return (x*(N - 1.f) + 0.5f)/N ;
}



#endif

