#pragma once
/**
qcurand.h -  Type dispatchers mapping T to cuRAND functions and Exponential generator
=======================================================================================

This aims to provide a CUDA equivalent G4RandExponential::shoot which is CLHEP::RandExponential::shoot::

    double RandExponential::shoot() {
      return -std::log(HepRandom::getTheEngine()->flat());
    }

    double RandExponential::shoot(double mean) {
      return -std::log(HepRandom::getTheEngine()->flat())*mean;
    }


Actually need to accept mean argument::

    // Calculate time delay (Non-Radiative + Intrinsic Emission)
    G4double deltaTime = 0.;
    if (tau_nr > 0.0)  deltaTime += G4RandExponential::shoot(tau_nr);
    if (tau_rad > 0.0) deltaTime += G4RandExponential::shoot(tau_rad);


* https://gitlab.cern.ch/CLHEP/CLHEP/-/blob/develop/Random/src/RandExponential.cc




**/

#if defined(__CUDACC__)
#include "qlog.h"
#include <curand_kernel.h>

template <typename T>
__device__ inline T      qcurand_uniform(curandState_t* state);

template <>
__device__ inline float  qcurand_uniform<float>(curandState_t* state) {  return curand_uniform(state); }

template <>
__device__ inline double qcurand_uniform<double>(curandState_t* state) { return curand_uniform_double(state); }

template <typename T>
__device__ inline T      qcurand_shoot_exponential(curandState_t* state) {         return -qlog(qcurand_uniform<T>(state)) ; }

template <typename T>
__device__ inline T      qcurand_shoot_exponential(curandState_t* state, T mean) { return -qlog(qcurand_uniform<T>(state))*mean ; }


#endif // __CUDACC__

