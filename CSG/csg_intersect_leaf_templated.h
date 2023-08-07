#pragma once

/**
csg_intersect_leaf_templated.h : distance_leaf and intersect_leaf functions
=============================================================================

**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define LEAF_FUNC __forceinline__ __device__
#else
#    define LEAF_FUNC inline
#endif


#define RT_DEFAULT_MAX 1.e27f

#if defined(__CUDACC__)
#include "math_constants.h"
#else

union uif_t 
{
    unsigned u ; 
    int i ; 
    float f ; 
};

LEAF_FUNC
float __int_as_float(int i)
{
    uif_t uif ; 
    uif.i = i ; 
    return uif.f ; 
}

#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_PI_F             3.141592654f

#endif


#include "OpticksCSG.h"
#include "squad.h"
#include "scuda_templated.h"

#include "CSGNode.h"
#include "CSGPrim.h"

#include "csg_robust_quadratic_roots.h"
#include "csg_classify.h"

#ifdef DEBUG_RECORD
#include <csignal>
#endif

#ifdef DEBUG_CYLINDER
#include "CSGDebug_Cylinder.hh"
#endif


template<typename F>
LEAF_FUNC F distance_leaf_sphere(const F3& pos, const Q4& q0 )
{
    F3 p ;
    p.x = pos.x - q0.f.x ; 
    p.y = pos.y - q0.f.y ; 
    p.z = pos.z - q0.f.z ; 
    F radius = q0.f.w;
    F sd = length(p) - radius ; 
    return sd ; 
}


