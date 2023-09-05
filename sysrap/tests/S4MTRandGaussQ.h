#pragma once
/**
S4MTRandGaussQ.h 
==================

Experiments using Geant4 extracts from below classes::

   g4-cls G4MTRandGaussQ
   g4-cls RandGaussQ

In order to work out how to do "G4RandGauss::shoot(0.0,sigma_alpha)"
GPU side. 

The transformQuick/transformSmall looks like a mapping 
of InverseErf onto (0,1) rather than (-1, 1) 

So maybe can use the CUDA func::

    u = curand_uniform(&rng) ;   // u uniform random in 0 to 1 
    u2 = 2.f*u - 1.f ;           // u2 uniform random in -1 to 1  
    float v = erfinvf(u2)


* https://mathworld.wolfram.com/InverseErf.html
* https://mathworld.wolfram.com/InverseErfc.html
* https://en.wikipedia.org/wiki/Error_function
* https://en.wikipedia.org/wiki/Q-function


__device__ float erfcf ( float  x )
    Calculate the complementary error function of the input argument. 

__device__ float erfcinvf ( float  x )
    Calculate the inverse complementary error function of the input argument. 

    * https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g31faaaeab2a785191c3e0e66e030ceca

    erfinvf(+-0) returns 0

    erfinvf(+1)  return +inf

    erfinvf(-1) returns -inf



__device__ float erfcxf ( float  x )
    Calculate the scaled complementary error function of the input argument. 

__device__ float erff ( float  x )
    Calculate the error function of the input argument. 

__device__ float erfinvf ( float  x )
    Calculate the inverse error function of the input argument. 


**/

struct S4MTRandGaussQ
{
  static double transformQuick (double r); 
  static double transformSmall (double r); 
}; 




