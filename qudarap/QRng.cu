/**
See /Developer/NVIDIA/CUDA-9.1/samples/0_Simple/cppIntegration/cppIntegration.cu
**/

#include "curand_kernel.h"


__global__ void QRng_generate_kernel(int threads_per_launch, curandState* rng_states, float* d_out )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    float u = curand_uniform(&rng_states[id]); 
    d_out[id] = u ;   
}


extern "C" void QRng_generate(int threads_per_launch, curandState* rng_states, float* d_out )
{
    dim3 grid(1, 1, 1);
    dim3 threads(threads_per_launch, 1, 1); 

    QRng_generate_kernel<<< grid, threads >>>( threads_per_launch, rng_states, d_out  );
} 
