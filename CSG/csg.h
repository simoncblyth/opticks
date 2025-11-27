#pragma once

#if defined(__CUDACC__)
    #define csg_HD __host__ __device__
    #define csg_D  __device__
    #define csg_G  __global__
    #define csg_C __constant__
#else
    #define csg_HD
    #define csg_D
    #define csg_G
    #define csg_C
#endif

