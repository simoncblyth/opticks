// name=basic_test ; gcc $name.cc -std=c++11 -lstdc++ -I/usr/local/cuda/include -DWITH_THRUST -o /tmp/$name && /tmp/$name
// name=basic_test ; gcc $name.cc -std=c++11 -lstdc++ -I/usr/local/cuda/include               -o /tmp/$name && /tmp/$name

/**

https://proofwiki.org/wiki/Sine_Exponential_Formulation

**/

#include <stdio.h>
#include <math_constants.h>

#ifdef WITH_THRUST
#include <thrust/complex.h>
using thrust::complex ; 
using thrust::sqrt ; 
using thrust::exp ; 
using thrust::sin ; 
using thrust::cos ; 
#else
#include <complex>
using std::complex ; 
using std::sqrt ; 
using std::exp ; 
using std::sin ; 
using std::cos ; 
#endif


int main()
{
    complex<float> i(0.f,1.f) ; 
    complex<float> z0 = i ; 
    complex<float> z1 = sqrt(z0)  ; 
    complex<float> z2 = exp(z0*CUDART_PI_F)  ; 
    complex<float> z3 = sin(z0)  ; 
    complex<float> z4 = cos(z0)  ; 

    printf("      z0 ( %10.4f %10.4f )                     \n", z0.real(), z0.imag() );  
    printf("      z1 ( %10.4f %10.4f ) sqrt(z0)            \n", z1.real(), z1.imag() );  
    printf("      z2 ( %10.4f %10.4f ) exp(z0*CUDART_PI_F) \n", z2.real(), z2.imag() );  
    printf("      z3 ( %10.4f %10.4f ) sin(z0)             \n", z3.real(), z3.imag() );  
    printf("      z4 ( %10.4f %10.4f ) cos(z0)             \n", z4.real(), z4.imag() );  

    return 0 ; 
}




