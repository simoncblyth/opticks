#pragma once


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SCUDA_HOSTDEVICE __host__ __device__
#    define SCUDA_INLINE __forceinline__
#else
#    define SCUDA_HOSTDEVICE
#    define SCUDA_INLINE inline
#endif


SCUDA_INLINE SCUDA_HOSTDEVICE double length(const double2& v)
{
    return sqrtf(dot(v, v));
}




