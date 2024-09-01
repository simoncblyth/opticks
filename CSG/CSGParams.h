#pragma once

/**
CSGParams.h
==============

Aiming to do CUDA only CSG intersect tests that follow 
the pattern of OptiX ray trace tests. 

CSGOptiX7.cu revolves around the Params constant::

    extern "C" { __constant__ Params params ;  }

Try to do something similar here for CUDA CSG scanning.

**/


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
    const quad4*     qq ;   // "query" rays
    quad4*           tt ;   // intersects 
    int             num ; 

    void intersect( int idx ); 
}; 

inline void CSGParams::intersect( int idx )
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

