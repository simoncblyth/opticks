#ifndef INIT_RNG_H
#define INIT_RNG_H

void init_rng_wrapper(void* dev_rng_states, unsigned int size, unsigned int threads_per_block, unsigned long long seed, unsigned long long offset);

#endif
