#include <stdio.h>
#include "qprop.h"

__global__ void _QProp_lookup( qprop* prop, float* lookup , const float* domain , unsigned lookup_prop, unsigned domain_width )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= domain_width || iy >= lookup_prop  ) return;

    float x = domain[ix] ; 
    float y = prop->interpolate( iy, x ); 
    unsigned index = iy * domain_width + ix ;

    if( iy == 0 )
    printf("//_QProp_lookup ix %3d x %10.4f  iy %d  y %10.4f prop->width %3d prop->height %3d \n", ix, x, iy, y, prop->width, prop->height ); 

    lookup[index] = y ; 
}

extern "C" void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qprop* prop, 
    float* lookup, 
    const float* domain, 
    unsigned lookup_prop, 
    unsigned domain_width
)
{
    _QProp_lookup<<<numBlocks,threadsPerBlock>>>( prop, lookup, domain, lookup_prop, domain_width ) ;
} 

