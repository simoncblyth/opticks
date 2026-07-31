#pragma once

/**
scie.h
========

http://www.ppsloan.org/publications/XYZJCGT.pdf

Simple Analytic Approximations to the CIE XYZ Color Matching Functions
Chris Wyman, Peter-Pike Sloan, Peter Shirley
NVIDIA

**/

#include "scuda.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SCIE_METHOD __host__ __device__
#else
#    define SCIE_METHOD
#endif



struct scie
{
    static SCIE_METHOD float xFit_1931( float wave );
    static SCIE_METHOD float yFit_1931( float wave );
    static SCIE_METHOD float zFit_1931( float wave );
    static SCIE_METHOD float3 xyzFit_1931(float wave);
};












inline float scie::xFit_1931( float wave )
{
    float t1 = (wave-442.0f)*((wave<442.0f)?0.0624f:0.0374f);
    float t2 = (wave-599.8f)*((wave<599.8f)?0.0264f:0.0323f);
    float t3 = (wave-501.1f)*((wave<501.1f)?0.0490f:0.0382f);
    return 0.362f*expf(-0.5f*t1*t1) + 1.056f*expf(-0.5f*t2*t2)- 0.065f*expf(-0.5f*t3*t3);
}

inline float scie::yFit_1931( float wave )
{
    float t1 = (wave-568.8f)*((wave<568.8f)?0.0213f:0.0247f);
    float t2 = (wave-530.9f)*((wave<530.9f)?0.0613f:0.0322f);
    return 0.821f*expf(-0.5f*t1*t1) + 0.286f*expf(-0.5f*t2*t2);
}

inline float scie::zFit_1931( float wave )
{
    float t1 = (wave-437.0f)*((wave<437.0f)?0.0845f:0.0278f);
    float t2 = (wave-459.0f)*((wave<459.0f)?0.0385f:0.0725f);
    return 1.217f*expf(-0.5f*t1*t1) + 0.681f*expf(-0.5f*t2*t2);
}

inline float3 scie::xyzFit_1931(float wave)
{
    float x = xFit_1931(wave);
    float y = yFit_1931(wave);
    float z = zFit_1931(wave);
    float3 xyz = make_float3( x, y, z ) ;
    return xyz ;
}


