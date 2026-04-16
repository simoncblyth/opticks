

#include "curand_kernel.h"
using Philox = curandStatePhilox4_32_10 ;
using RNG = Philox ;

#include "scuda.h"
#include "squad.h"

#include "qscint_three.h"

#include "stdio.h"


__global__ void _QScintThree_wavelength(qscint_three* scint, float* wavelength, unsigned num_wavelength, int species )
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    if (ix >= num_wavelength) return;

    RNG rng ;
    {
        unsigned long long subsequence = ix ;
        unsigned long long seed = 0ull ;
        unsigned long long offset = 0ull ;
        curand_init( seed, subsequence, offset, &rng );
    }

    float u = curand_uniform(&rng);
    float w = scint->wavelength_hd20(species, u) ;

    if(ix % 100000 == 0) printf("//_QScintThree_wavelength ix %d  w %10.4f    \n", ix, w  );

    wavelength[ix] = w ;
}

extern "C" void QScintThree_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qscint_three* scint, float* wavelength, unsigned num_wavelength, int species )
{
    printf("//QScintThree_wavelength num_wavelength %d species %d \n", num_wavelength, species );
    _QScintThree_wavelength<<<numBlocks,threadsPerBlock>>>( scint, wavelength, num_wavelength, species );
}




