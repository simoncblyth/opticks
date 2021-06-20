
#include "curand_kernel.h"

__global__ void _QScint_generate_kernel(curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    float u = curand_uniform(&rng_states[id]); 

    float wl = tex2D<float>(texObj,  u, 0.f);

    wavelength[id] = wl ;   
}

extern "C" void QScint_generate_kernel(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength ) 
{
    _QScint_generate_kernel<<<numBlocks,threadsPerBlock>>>( rng_states, texObj, wavelength, num_wavelength );
} 


