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
    float wl = scint->wavelength_hd20(species, u0) ;

    //bool dump = ix % 100000 == 0 ;
    //bool dump = wl < 200.f ;
    bool dump = u0 == 1.f || u0 == 0.f ;
    //bool dump = u0 > 0.99999f ;

    if(dump) printf("//_QScintThree_wavelength ix %d iy %d nx %d ny %d  u0 %10.8f wl %10.4f\n", ix, iy, nx, ny, u0, wl);
    wavelength[index] = wl;
}

extern "C" void QScintThree_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qscint_three* scint, float* wavelength, size_t width, size_t height, bool hd)
{
    printf("//QScintThree_wavelength width %d height %d hd %d \n", width, height, hd);
    _QScintThree_wavelength<<<numBlocks, threadsPerBlock>>>(scint, wavelength, width, height, hd);
}

extern "C" int QScintThree_wavelength_WITH_LERP()
{
#ifdef WITH_LERP
    return 1;
#else
    return 0;
#endif
}






