#ifndef CURANDWRAPPER_KERNEL_H
#define CURANDWRAPPER_KERNEL_H

#include "cuda.h"
#include "curand_kernel.h"

class LaunchSequence ; 

void init_rng_wrapper( LaunchSequence* launchseq, CUdeviceptr dev_rng_states, unsigned long long seed, unsigned long long offset);
void test_rng_wrapper( LaunchSequence* launchseq, CUdeviceptr dev_rng_states, float* host_a);
curandState* copytohost_rng_wrapper( LaunchSequence* launchseq, CUdeviceptr dev_rng_states);

CUdeviceptr allocate_rng_wrapper( LaunchSequence* launchseq);
void free_rng_wrapper( CUdeviceptr dev_rng_states );

CUdeviceptr copytodevice_rng_wrapper( LaunchSequence* launchseq, void* host_rng_states);

void devicesync_wrapper();


#endif
