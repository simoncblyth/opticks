#pragma once
/**
basic_complex.h
==================

Exercising complex number arithmetic with common source used on device and host. 

**/

#include <stdio.h>
#include <math_constants.h>

#ifdef WITH_THRUST
#include <thrust/complex.h>
#else
#include <complex>
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define BASIC_METHOD __host__ __device__ __forceinline__
#else
#    define BASIC_METHOD inline 
#endif

namespace Const
{
    template<typename T> 
    BASIC_METHOD constexpr T zero(){ return T(0.0) ; }
 
    template<typename T>
    BASIC_METHOD constexpr T one() { return T(1.0) ; }

    template<typename T>
    BASIC_METHOD constexpr T two() { return T(2.0) ; }
     
    template<typename T>
    BASIC_METHOD constexpr T pi() { return T(CUDART_PI) ; }

    template<typename T>
    BASIC_METHOD constexpr T twopi() { return T(2.0*CUDART_PI) ; }
}

// constexpr should mean that the calc of the above 
// functions happen at compile time,  not when they are used


/**
Need the BASIC_METHOD above to avoid::

    warning: calling a constexpr __host__ function("zero") from a __host__
    __device__ function("test") is not allowed. The experimental flag
    '--expt-relaxed-constexpr' can be used to allow this.

**/


template <typename T>
struct basic_complex
{
    BASIC_METHOD static void test() ; 
}; 

template <typename T>
BASIC_METHOD void basic_complex<T>::test()
{
#ifdef WITH_THRUST
    using thrust::complex ; 
    using thrust::sqrt ; 
    using thrust::exp ; 
    using thrust::sin ; 
    using thrust::cos ; 
#else
    using std::complex ; 
    using std::sqrt ; 
    using std::exp ; 
    using std::sin ; 
    using std::cos ; 
#endif


    complex<T> i(Const::zero<T>(),Const::one<T>()) ; 
    complex<T> z0 = i ; 
    complex<T> z1 = sqrt(z0)  ; 
    complex<T> z2 = exp(z0*Const::pi<T>())  ; 
    complex<T> z3 = sin(z0)  ; 
    complex<T> z4 = cos(z0)  ; 

    printf("//basic_complex::test\n"); 
    printf("      z0 ( %10.4f %10.4f )                     \n", z0.real(), z0.imag() );  
    printf("      z1 ( %10.4f %10.4f ) sqrt(z0)            \n", z1.real(), z1.imag() );  
    printf("      z2 ( %10.4f %10.4f ) exp(z0*CUDART_PI_F) \n", z2.real(), z2.imag() );  
    printf("      z3 ( %10.4f %10.4f ) sin(z0)             \n", z3.real(), z3.imag() );  
    printf("      z4 ( %10.4f %10.4f ) cos(z0)             \n", z4.real(), z4.imag() );  

}


