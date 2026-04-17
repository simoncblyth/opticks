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
    QSCINT_METHOD float   wavelength_hd20(int species_idx, float u0) const ;
#endif

};


#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CURAND) || defined(MOCK_CUDA)
inline QSCINT_METHOD float qscint_three::wavelength_sd(int species_idx, float u0) const
{
    // dont use the HD layers
    int layer = (species_idx * 3) ;
    return tex2DLayered<float>(scint_tex_layered, u0, 0.5f, layer);
}

/**
qscint_three::wavelength_hd20
-------------------------------

NB this _hd20 method is tied to the form and layout of the texture scint_tex_layered
and its input array prepared by U4ScintCommon::CreateGeant4InterpolatedInverseCDF
as instructed from U4ScintThree::U4ScintThree.
Due to this it does not make sense to have a _hd10 method too,
as would need a separate tex for that.
Because the _sd simply uses the full range layer there is no problem with that one.

**/

inline QSCINT_METHOD float qscint_three::wavelength_hd20(int species_idx, float u0) const
{
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

#endif

