#pragma once
/**
s_mock_erfcinvf.h
==================

This is only intended to be included for MOCK_CUDA on CPU running, 
it defines a global function to standin for the CUDA equivalent.  

https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g47e42e1bbdda4a98b07fbca5e2a5c396

__device__ float erfcinvf ( float  x )
    Calculate the inverse complementary error function of the input argument.

    Calculate the inverse complementary error function
    (x), of the input argument x in the interval [0, 2]. 

**/

#include "njuffa_erfcinvf.h"
float erfcinvf(float u2){  return njuffa_erfcinvf(u2) ; }



