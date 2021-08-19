#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define CSG_FUNC __forceinline__ __device__ __host__
#else
#    define CSG_FUNC inline
#endif


/**2
csg_intersect_boolean.h : struct CSG
-------------------------------------------

Small stack of float4 isect (surface normals and t:distance_to_intersect).

csg_push 
    push float4 isect and nodeIdx into stack

csg_pop
    pop float4 isect and nodeIdx off the stack   

2**/

#define CSG_STACK_SIZE 15

struct CSG_Stack 
{
   float4 data[CSG_STACK_SIZE] ; 
   unsigned idx[CSG_STACK_SIZE] ; 
   int curr ;
};

CSG_FUNC 
int csg_push(CSG_Stack& csg, const float4& isect, unsigned nodeIdx)
{
    if(csg.curr >= CSG_STACK_SIZE - 1) return ERROR_OVERFLOW ; 
    csg.curr++ ; 
    csg.data[csg.curr] = isect ; 
    csg.idx[csg.curr] = nodeIdx ; 
    return 0 ; 
}
CSG_FUNC 
int csg_pop(CSG_Stack& csg, float4& isect, unsigned& nodeIdx)
{
    if(csg.curr < 0) return ERROR_POP_EMPTY ;     
    isect = csg.data[csg.curr] ;
    nodeIdx = csg.idx[csg.curr] ;
    csg.idx[csg.curr] = 0u ;   // scrub the idx for debug
    csg.curr-- ; 
    return 0 ; 
}

CSG_FUNC 
int csg_pop0(CSG_Stack& csg)   // pop without returning anything 
{
    if(csg.curr < 0) return ERROR_POP_EMPTY ;     
    csg.idx[csg.curr] = 0u ;   // scrub the idx for debug
    csg.curr-- ; 
    return 0 ; 
}

CSG_FUNC 
unsigned long long csg_repr(CSG_Stack& csg)
{
    unsigned long long val = 0 ; 
    if(csg.curr == -1) return val ; 

    unsigned long long c = csg.curr ; 
    val |= (c+1ull) ;  // count at lsb, contents from msb 
 
    do { 
        unsigned long long x = csg.idx[c] & 0xf ;
        val |=  x << ((16ull-c-1ull)*4ull) ; 
    } 
    while(c--) ; 

    return val ; 
} 



