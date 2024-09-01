#pragma once

/**
CSGParams.h : HMM: needs better name : purpose is for scan testing
======================================================================

See CSGScan.cc for usage

Aiming to do CUDA only CSG intersect tests that follow 
the pattern of OptiX ray trace tests. 

CSGOptiX7.cu revolves around the Params constant::

    extern "C" { __constant__ Params params ;  }

Try to do something similar here for CUDA CSG scanning.

**/



#if defined(__CUDACC__) || defined(__CUDABE__)
   #define PARAMS_METHOD __device__
#else
   #define PARAMS_METHOD 
#endif 



#include "scuda.h"
#include "sqat4.h"
#include "squad.h"

#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"


struct CSGParams
{
    const CSGNode* node ; 
    const float4*  plan ;
    const qat4*    itra ; 
    quad4*           qq ;   // "query" rays
    quad4*           tt ;   // intersects 
    int             num ; 
    bool           devp ;   // device pointers

    PARAMS_METHOD void intersect( int idx ); 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    PARAMS_METHOD int num_valid_isect();
#endif

}; 

inline PARAMS_METHOD void CSGParams::intersect( int idx )
{
    const quad4* q = qq + idx ; 
    const float t_min = q->q1.f.w ; 
    const float3* ori = q->v0(); 
    const float3* dir = q->v1(); 

    float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
    bool valid_isect = intersect_prim(isect, node, plan, itra, t_min, *ori, *dir );

    quad4* t = tt + idx ; 

    *t = *q ; 

    t->q0.i.w = int(valid_isect) ;

    if( valid_isect )
    {
        t->q2.f.x  = ori->x + isect.w * dir->x ;   
        t->q2.f.y  = ori->y + isect.w * dir->y ;   
        t->q2.f.z  = ori->z + isect.w * dir->z ;   

        t->q3.f    = isect ;  
    }
} 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline PARAMS_METHOD int CSGParams::num_valid_isect()
{
    int n_hit = 0 ; 
    for(int i=0 ; i < num ; i++)
    {
        const quad4& t = tt[i] ;
        bool hit = t.q0.i.w == 1 ; 
        if(hit)  n_hit += 1 ; 
    }
    return n_hit ; 
} 
#endif


