#pragma once
/**
s_mock_curand.h
=================

s_mock_curand provides a wrapper for the random 
generation of srngcpu that allows CUDA intended code
to be tricked into running on CPU. 

This is conditionally included by scurand.h 

Mocking *curand_uniform* enables code developed for use 
with the standard CUDA *curand_uniform* to be tested on CPU without change, 
other than switching headers. 

Note that instanciation API does not match the real one, 
but that does that matter as instanciation doesnt need testing. 

**/

#include "srngcpu.h"

typedef srngcpu curandStateXORWOW ; 
typedef srngcpu curandState_t ; 
//
inline float curand_uniform(curandState_t* state ){         return state->generate_float() ; }
inline double curand_uniform_double(curandState_t* state ){ return state->generate_double() ; }



