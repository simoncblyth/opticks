#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qcurand.h"
#include "qrng.h"

/**
_QRng_generate
-----------------

Simple curand generation with skipahead, no encapsulation. 

* ni: number of items, eg photons
* nv: number of values to generate for each item 

**/


template <typename T>
__global__ void _QRng_generate(T* uu, unsigned ni, unsigned nv, curandStateXORWOW* r, unsigned long long skipahead_  )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= ni) return;

    curandState rng = *(r + id) ; 
    skipahead( skipahead_, &rng ); 

    unsigned ibase = id*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        T u = qcurand<T>::uniform(&rng) ;

        //if( id == 0 ) printf("//_QRng_generate id %d v %d u %10.4f  skipahead %d \n", id, v, u, skipahead_  ); 

        uu[ibase+v] = u ; 
    }
}

template <typename T>
extern void QRng_generate(dim3 numBlocks, dim3 threadsPerBlock, T* uu, unsigned ni, unsigned nv, curandStateXORWOW* r, unsigned long long skipahead_ )
{
    printf("//QRng_generate ni %d nv %d skipahead %llu \n", ni, nv, skipahead_ ); 
    _QRng_generate<T><<<numBlocks,threadsPerBlock>>>( uu, ni, nv, r, skipahead_ );
} 


template void QRng_generate(dim3, dim3, float*, unsigned, unsigned, curandStateXORWOW*, unsigned long long ); 
template void QRng_generate(dim3, dim3, double*, unsigned, unsigned, curandStateXORWOW*, unsigned long long ); 



/**
_QRng_generate_2
--------------------

Try with a little bit of encapsulation into qrng. 
Would moving more into qrng be appropriate ?

Using light touch encapsulation of setup only as want generation 
of randoms to be familiar/standard and suffer no overheads.

**/

template <typename T>
__global__ void _QRng_generate_2(qrng* qr, unsigned event_idx, T* uu, unsigned ni, unsigned nv )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= ni) return;

    curandState rng ; 
    qr->random_setup(rng, event_idx, id ); 

    unsigned ibase = id*nv ; 

    for(unsigned v=0 ; v < nv ; v++)
    {
        T u = qcurand<T>::uniform(&rng) ;

        //if( id == 0 ) printf("//_QRng_generate_2 id %d v %d u %10.4f \n", id, v, u  ); 

        uu[ibase+v] = u ; 
    }
}

template <typename T>
extern void QRng_generate_2(dim3 numBlocks, dim3 threadsPerBlock, qrng* qr, unsigned event_idx, T* uu, unsigned ni, unsigned nv )
{
    printf("//QRng_generate_2 event_idx %d ni %d nv %d \n", event_idx, ni, nv ); 
    _QRng_generate_2<T><<<numBlocks,threadsPerBlock>>>( qr, event_idx, uu, ni, nv );
} 

template void QRng_generate_2(dim3, dim3, qrng*, unsigned, float*,  unsigned, unsigned ); 
template void QRng_generate_2(dim3, dim3, qrng*, unsigned, double*, unsigned, unsigned ); 


