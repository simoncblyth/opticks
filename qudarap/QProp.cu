#include <stdio.h>
#include "qprop.h"

__global__ void _QProp_lookup( qprop* prop, float* lookup , const float* domain , unsigned iprop, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= domain_width ) return;

    float x = domain[ix] ; 
    float y = prop->interpolate( iprop, x ); 
    unsigned index = iprop * domain_width + ix ;

    if( iprop == 0 )
    printf("//_QProp_lookup ix %3d x %10.4f  iprop %d  y %10.4f prop->width %3d prop->height %3d \n", ix, x, iprop, y, prop->width, prop->height ); 

    lookup[index] = y ; 
}

extern "C" void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qprop* prop, 
    float* lookup, 
    const float* domain, 
    unsigned iprop, 
    unsigned domain_width
)
{
    _QProp_lookup<<<numBlocks,threadsPerBlock>>>( prop, lookup, domain, iprop, domain_width ) ;
} 

