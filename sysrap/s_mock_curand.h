#pragma once
/**
s_mock_curand.h
=================

This is conditionally included by scurand.h 


Mocking *curand_uniform* enables code developed for use 
with the standard CUDA *curand_uniform* to be tested on CPU without change, 
other than switching headers. 

TODO: 
   provide option to hook into the precooked randoms (see SRngSpec)
   so the "generated" values actually match curand_uniform on device  
   
   * this requires setting an index like OpticksRandom.hh does 

HMM: instanciation API does not match the real one, does that matter ?

**/

#include "srng.h"

//typedef curandStateXORWOW curandState_t ; 
typedef srng curandStateXORWOW ; 
typedef srng curandState_t ; 

float curand_uniform(curandState_t* state ){         return state->generate_float() ; }
double curand_uniform_double(curandState_t* state ){ return state->generate_double() ; }



