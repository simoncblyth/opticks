#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define TRANCHE_FUNC __forceinline__ __device__
#else
#    define TRANCHE_FUNC inline
#endif




/**1
csg_intersect_boolean.h : struct Tranche
-------------------------------------------

Postorder Tranch storing a stack of slices into the postorder sequence.

slice
   32 bit unsigned holding a pair of begin and end indices 
   specifying a range over the postorder traversal sequence

tranche_push 
    push (slice, tmin) onto the small stack 

tranche_pop
    pops off (slice, tmin) 

tranche_repr
    representation of the stack of slices packed into a 64 bit unsigned long long  

1**/

#define TRANCHE_STACK_SIZE 4


struct Tranche
{
    float      tmin[TRANCHE_STACK_SIZE] ;  
    unsigned  slice[TRANCHE_STACK_SIZE] ; 
    int curr ;
};


TRANCHE_FUNC
int tranche_push(Tranche& tr, const unsigned slice, const float tmin)
{
    if(tr.curr >= TRANCHE_STACK_SIZE - 1) return ERROR_TRANCHE_OVERFLOW ; 
    tr.curr++ ; 
    tr.slice[tr.curr] = slice  ; 
    tr.tmin[tr.curr] = tmin ; 
    return 0 ; 
}

TRANCHE_FUNC
int tranche_pop(Tranche& tr, unsigned& slice, float& tmin)
{
    if(tr.curr < 0) return ERROR_POP_EMPTY  ; 
    slice = tr.slice[tr.curr] ;
    tmin = tr.tmin[tr.curr] ;  
    tr.curr-- ; 
    return 0 ; 
}

TRANCHE_FUNC
unsigned long long tranche_repr(Tranche& tr)
{
    unsigned long long val = 0ull ; 
    if(tr.curr == -1) return val ; 

    unsigned long long c = tr.curr ;
    val |= ((c+1ull)&0xfull)  ;     // count at lsb, contents from msb 
 
    do { 
        unsigned long long x = tr.slice[c] & 0xffff ;
        val |=  x << ((4ull-c-1ull)*16ull) ; 
    } 
    while(c--) ; 

    return val ; 
} 



