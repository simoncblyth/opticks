/**
See /Developer/NVIDIA/CUDA-9.1/samples/0_Simple/cppIntegration/cppIntegration.cu
**/

#include "curand_kernel.h"


__global__ void _QGen_generate_kernel(curandState* rng_states, float* out, unsigned num_gen )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_gen) return;

    float u = curand_uniform(&rng_states[id]); 
    out[id] = u ;   
}

extern "C" void QGen_generate_kernel(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, float* out, unsigned num_gen ) 
{
    _QGen_generate_kernel<<< numBlocks, threadsPerBlock >>>( rng_states, out, num_gen  );
} 
