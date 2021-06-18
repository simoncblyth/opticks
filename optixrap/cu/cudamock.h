#pragma once


#include "sutil_vec_math.h"
#include <cmath>
#include <cstdio>
#include "curandState.h" // CPU MOCKING  


int      __float2int_rn(float f){  return static_cast<int>(f) ; }  // PLACEHOLDER : NOT THE SAME ROUNDING 
unsigned __float2uint_rn(float f){ return static_cast<unsigned>(f) ; }  // PLACEHOLDER : NOT THE SAME ROUNDING 
unsigned __ffs(unsigned u){  return __builtin_ffs(u) ; }  // ?

