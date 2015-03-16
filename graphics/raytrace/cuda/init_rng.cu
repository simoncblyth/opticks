#include "init_rng.hh"

#include "curand_kernel.h"
#include "stdio.h"


#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)

#define CUT_CHECK_ERROR(errorMessage) do {                                 \
    cudaThreadSynchronize();                                                \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)




__global__ void init_rng(int nthreads, curandState* rng_states, unsigned long long seed, unsigned long long offset)
{
   int id = blockIdx.x*blockDim.x + threadIdx.x;
   if (id >= nthreads) return;

   curand_init(seed, id, offset, &rng_states[id]);
}




void init_rng_wrapper(void* dev_rng_states, unsigned int size, unsigned int threads_per_block, unsigned long long seed, unsigned long long offset)
{
    unsigned int nthreads = size ;
    unsigned int blocks_per_grid = nthreads/threads_per_block + 1 ;

    printf("init_rng_wrapper  nthreads %u blocks_per_grid %u threads_per_block %u seed %llu offset %llu \n", 
          nthreads, 
          blocks_per_grid, 
          threads_per_block, 
          seed, 
          offset); 

    init_rng<<<blocks_per_grid, threads_per_block>>>(nthreads, (curandState*)dev_rng_states, seed, offset );

    printf("init_rng_wrapper kernel call completed\n");

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    printf("init_rng_wrapper post kernel call Sync completed\n");
}



