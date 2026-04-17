#include "curand_kernel.h"
using Philox = curandStatePhilox4_32_10;
using RNG = Philox;
#include "scuda.h"
#include "squad.h"
#include "qscint_three.h"
#include "stdio.h"


__global__ void _QScintThree_wavelength(qscint_three* scint, float* wavelength, size_t width, size_t height, bool hd )
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned nx = width;
    unsigned ny = height;
    if (ix >= nx) return;
    if (iy >= ny) return;
    unsigned index = iy*nx + ix;
    RNG rng;
    {
        unsigned long long subsequence = ix;
        unsigned long long seed = 0ull;
        unsigned long long offset = 0ull;
        curand_init(seed, subsequence, offset, &rng);
    }
    int species = iy;
    float u0 = curand_uniform(&rng);
    float wl = hd ? scint->wavelength_hd20(species, u0) : scint->wavelength_sd(species, u0) ;
    if(ix % 100000 == 0) printf("//_QScintThree_wavelength ix %d iy %d  w %10.4f    \n", ix, iy, wl);
    wavelength[index] = wl;
}

extern "C" void QScintThree_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qscint_three* scint, float* wavelength, size_t width, size_t height, bool hd)
{
    printf("//QScintThree_wavelength width %d height %d hd %d \n", width, height, hd);
    _QScintThree_wavelength<<<numBlocks, threadsPerBlock>>>(scint, wavelength, width, height, hd);
}


