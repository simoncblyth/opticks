#include "stdio.h"
#include "curand_kernel.h"
#include "scuda.h"
#include "qcurand.h"


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

        if( id == 0 )
        printf("//_QRng_generate id %d v %d u %10.4f  skipahead %d \n", id, v, u, skipahead_  ); 

        uu[ibase+v] = u ; 
    }
}

template <typename T>
extern void QRng_generate(dim3 numBlocks, dim3 threadsPerBlock, T* uu, unsigned ni, unsigned nv, curandStateXORWOW* r, unsigned long long skipahead_ )
{
    printf("//QRng_generate ni %d \n", ni ); 
    _QRng_generate<T><<<numBlocks,threadsPerBlock>>>( uu, ni, nv, r, skipahead_ );
} 


template void QRng_generate(dim3, dim3, float*, unsigned, unsigned, curandStateXORWOW*, unsigned long long ); 
template void QRng_generate(dim3, dim3, double*, unsigned, unsigned, curandStateXORWOW*, unsigned long long ); 


