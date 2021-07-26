#include "QUDARAP_API_EXPORT.hh"
#include <stdio.h>
#include "qprop.h"


template <typename T>
__global__ void _QProp_lookup( qprop<T>* prop, T* lookup , const T* domain , unsigned iprop, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;

    T x = domain[ix] ; 
    T y = prop->interpolate( iprop, x ); 
    unsigned index = iprop * domain_width + ix ;

    if( iprop == 0 )
    printf("//_QProp_lookup ix %3d x %10.4f  iprop %d  y %10.4f prop->width %3d prop->height %3d \n", ix, x, iprop, y, prop->width, prop->height ); 

    lookup[index] = y ; 
}

// NB this cannot be extern "C" as need C++ name mangling for template types

template <typename T> extern void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qprop<T>* prop, 
    T* lookup, 
    const T* domain, 
    unsigned iprop, 
    unsigned domain_width
)
{
    _QProp_lookup<T><<<numBlocks,threadsPerBlock>>>( prop, lookup, domain, iprop, domain_width ) ;
} 



template void QProp_lookup(dim3, dim3, qprop<double>*, double*, double const*, unsigned int, unsigned int) ; 
template void QProp_lookup(dim3, dim3, qprop<float>*, float*, float const*, unsigned int, unsigned int) ; 



