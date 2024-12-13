#include "stdio.h"

#include "scuda.h"
#include "qrng.h"
#include "scurand.h"


/**
_QRng_generate
--------------------

Try with a little bit of encapsulation into qrng. 
Would moving more into qrng be appropriate ?

Using light touch encapsulation of setup only as want generation 
of randoms to be familiar/standard and suffer no overheads.

**/


template <typename T>
__global__ void _QRng_generate(qrng<RNG>* qr, unsigned event_idx, T* uu, unsigned ni, unsigned nj )
{
    unsigned id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= ni) return;

    RNG rng ; 
    qr->init(rng, event_idx, id ); 

    unsigned ibase = id*nj ; 

    for(unsigned j=0 ; j < nj ; j++)
    {
        T u = scurand<T>::uniform(&rng) ;

        //if( id == 0 ) printf("//_QRng_generate_2 id %d v %d u %10.4f \n", id, v, u  ); 

        uu[ibase+j] = u ; 
    }
}

template <typename T>
extern void QRng_generate(dim3 numBlocks, dim3 threadsPerBlock, qrng<RNG>* qr, unsigned event_idx, T* uu, unsigned ni, unsigned nj )
{
    printf("//QRng_generate event_idx %d ni %d nj %d \n", event_idx, ni, nj ); 
    _QRng_generate<T><<<numBlocks,threadsPerBlock>>>( qr, event_idx, uu, ni, nj );
} 

template void QRng_generate(dim3, dim3, qrng<RNG>*, unsigned, float*,  unsigned, unsigned ); 
template void QRng_generate(dim3, dim3, qrng<RNG>*, unsigned, double*, unsigned, unsigned ); 


