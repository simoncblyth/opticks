#pragma once
/**
s_mock_curand.h
=================

Mocking *curand_uniform* enables code developed for use 
with the standard CUDA *curand_uniform* to be tested on CPU without change, 
other than switching headers. 

TODO: 
   provide option to hook into the precooked randoms (see SRngSpec)
   so the "generated" values actually match curand_uniform on device  
   
   * this requires setting an index like OpticksRandom.hh does 

HMM: instanciation API does not match the real one, does that matter ?

**/

#include <random>

struct curandStateXORWOW
{
    std::mt19937_64 engine ;
    std::uniform_real_distribution<float>  dist ; 
    curandStateXORWOW(unsigned seed_) : dist(0,1) { engine.seed(seed_) ; }
    float generate(){ return dist(engine) ; } 
}; 

typedef curandStateXORWOW curandState_t ; 

float curand_uniform(curandState_t* state ){ return state->generate() ; }


