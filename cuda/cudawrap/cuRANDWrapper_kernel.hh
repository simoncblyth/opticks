#ifndef CURANDWRAPPER_KERNEL_H
#define CURANDWRAPPER_KERNEL_H

#include "curand_kernel.h"

class LaunchSequence ; 

void init_rng_wrapper( 
   LaunchSequence* launchseq, 
   void* dev_rng_states, 
   unsigned long long seed, 
   unsigned long long offset
);


void test_rng_wrapper( 
   LaunchSequence* launchseq, 
   void* dev_rng_states, 
   float* host_a
);


curandState* create_rng_wrapper(
    LaunchSequence* launchseq
);

curandState* copytohost_rng_wrapper(
    LaunchSequence* launchseq,
    void* dev_rng_states
);
 
curandState* copytodevice_rng_wrapper(
    LaunchSequence* launchseq,
    void* host_rng_states
);


#endif
