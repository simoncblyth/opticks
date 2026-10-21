#pragma once
/**
sfunc.h
========

Simple functions - separate from scuda.h

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SCLAMP_FUNCTION __host__ __device__ __forceinline__
#else
#    define SCLAMP_FUNCTION inline
#endif


template <typename T>
SCLAMP_FUNCTION T sclamp(T val, T lo, T hi)
{
    return val < lo ? lo : (hi < val ? hi : val);
}




