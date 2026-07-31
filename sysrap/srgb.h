#pragma once
/**
srgb.h
=======

To get to grips with color spaces see::

   ~/env/graphics/ciexyz/sRGB.py


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SRGB_METHOD __host__ __device__
#else
#    define SRGB_METHOD
#endif


#include "scuda.h"
#include <cmath>
#include <algorithm>

struct srgb
{
    // Apply standard sRGB companding (linear -> gamma-encoded)
    SRGB_METHOD static inline float gammaCorrect(float c)
    {
        c = fmaxf(0.0f, fminf(1.0f, c));
        return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
    }

    // Convert CIE XYZ (D65) -> Linear sRGB
    SRGB_METHOD static inline float3 xyz2linear(const float3& xyz)
    {
        float r =  3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z;
        float g = -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z;
        float b =  0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z;
        return make_float3(r, g, b);
    }

    // Convert spectral CIE XYZ directly to normalized gamma-corrected sRGB [0, 1]
    SRGB_METHOD static inline float3 xyz2rgb(const float3& xyz, bool normalizeLuminance = true)
    {
        float3 rgb = xyz2linear(xyz);

        // Desaturate out-of-gamut (negative) spectral values by adding white
        float min_val = fminf(rgb.x, fminf(rgb.y, rgb.z));
        if (min_val < 0.0f) {
            rgb.x -= min_val;
            rgb.y -= min_val;
            rgb.z -= min_val;
        }

        // Normalize energy/brightness so peak value fits in [0, 1]
        float max_val = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
        if (max_val > 0.0f && normalizeLuminance) {
            rgb.x /= max_val;
            rgb.y /= max_val;
            rgb.z /= max_val;
        }

        // Apply sRGB transfer function
        return make_float3(
            gammaCorrect(rgb.x),
            gammaCorrect(rgb.y),
            gammaCorrect(rgb.z)
        );
    }
};





