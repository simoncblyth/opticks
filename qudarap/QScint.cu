

#include "curand_kernel.h"
#include "scuda.h"
#include "qscint.h"

/**
Minimize the code here, as this "junction" code cannot be easily tested/mocked for use from multiple contexts.
Instead implement in simple headers for flexibility of usage and testing.

TODO

* simplify this using a context argument that collects all the common args : rng_states, texObj, photon,  etc..

**/

__global__ void _QScint_generate_wavelength(curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_wavelength) return;

    qscint_wavelength( wavelength, id, rng_states + id, texObj ); 
}

extern "C" void QScint_generate_wavelength(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength ) 
{
    _QScint_generate_wavelength<<<numBlocks,threadsPerBlock>>>( rng_states, texObj, wavelength, num_wavelength );
} 


__global__ void _QScint_generate_photon(curandState* rng_states, cudaTextureObject_t texObj, quad4* photon, unsigned num_photon )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= num_photon) return;

    qscint_photon( photon, id, rng_states + id, texObj ); 
}

extern "C" void QScint_generate_photon(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, quad4* photon, unsigned num_photon ) 
{
    _QScint_generate_photon<<<numBlocks,threadsPerBlock>>>( rng_states, texObj, photon, num_photon );
} 




