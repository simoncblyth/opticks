#pragma once
/**
srpn_tranche.h : stack to hold tmin and begin/end indices slicing into an RPN "tape"
======================================================================================

Following from CSG/csg_tranche.h

* *begin* index is first of the range eg 0
* *end* index is one beyond the range eg 3 for a simple CSG combination of two primitives


HMM : 16 bits for begin and end is not needed - 8 bits would be enough
if something else could profitably included into the stack it can be



Checking for spill to local memory
-----------------------------------

::

     nvcc -ptx -arch=sm_86 kernel.cu -o kernel.ptx
     grep .local kernel.ptx


**/



#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SRPN_TRANCHE_METHOD __forceinline__ __device__ __host__
#else
#    define SRPN_TRANCHE_METHOD inline
#endif


#include "srpn_error.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#include <iostream>
#include <iomanip>
#endif


#define SRPN_TRANCHE_STACK_SIZE 4

struct srpn_tranche
{
    // discrete scalers(not an array) allows the nvcc compiler to keep everything in registers
    //
    float    t0, t1, t2, t3;
    unsigned s0, s1, s2, s3;
    int      curr;

    SRPN_TRANCHE_METHOD srpn_tranche()
            :
            t0(0.f), t1(0.f), t2(0.f), t3(0.f),
            s0(0),   s1(0),   s2(0),   s3(0),
            curr(-1) {}

    SRPN_TRANCHE_METHOD int push(int begin,  int  end, float  tmin_);
    SRPN_TRANCHE_METHOD int pop( int& begin, int& end, float& tmin_);

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SRPN_TRANCHE_METHOD std::string desc() const ;
#endif

};


SRPN_TRANCHE_METHOD
int srpn_tranche::push(int begin, int end, float tmin_)
{
    if(curr >= SRPN_TRANCHE_STACK_SIZE - 1) return SRPN_ERROR_TRANCHE_OVERFLOW ;

    curr++ ;

    unsigned slice = (( begin & 0xffffu ) << 16 ) | ( end & 0xffffu ) ;

    switch(curr)
    {
        case 0: s0 = slice ; t0 = tmin_ ; break ;
        case 1: s1 = slice ; t1 = tmin_ ; break ;
        case 2: s2 = slice ; t2 = tmin_ ; break ;
        case 3: s3 = slice ; t3 = tmin_ ; break ;
        default:                          break ;
    }
    return 0 ;
}


SRPN_TRANCHE_METHOD
int srpn_tranche::pop( int& begin, int& end, float& tmin_)
{
    if (curr < 0) return SRPN_ERROR_POP_EMPTY;

    unsigned slice = 0;

    // Direct register load via switch-case
    switch (curr)
    {
        case 0: slice = s0; tmin_ = t0; break;
        case 1: slice = s1; tmin_ = t1; break;
        case 2: slice = s2; tmin_ = t2; break;
        case 3: slice = s3; tmin_ = t3; break;
        default: break;
    }

    begin = slice >> 16;
    end   = slice & 0xffffu;
    curr--;

    return 0 ;
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
SRPN_TRANCHE_METHOD std::string srpn_tranche::desc() const
{

    float    t[SRPN_TRANCHE_STACK_SIZE] = {t0, t1, t2, t3};
    unsigned s[SRPN_TRANCHE_STACK_SIZE] = {s0, s1, s2, s3};

    std::stringstream ss ;
    ss
       << "[srpn_tranche::desc "
       << " curr " << curr
       << "\n"
       ;

    for(int i=0 ; i <= curr ; i++) ss
       << " i " << std::setw(1) << i
       << " begin "  << std::setw(4) << ( s[i] >> 16 )
       << " end "    << std::setw(4) << ( s[i] & 0xffffu )
       << " tmin "   << std::setw(10) << std::fixed << std::setprecision(4) << t[i]
       << "\n"
       ;

    ss << "]srpn_tranche::desc\n" ;
    std::string str = ss.str() ;
    return str ;
}

#endif




