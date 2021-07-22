#include <stdio.h>
#include "qprop.h"

__global__ void _QProp_lookup( float* lookup , const float* domain , unsigned lookup_prop, unsigned domain_width, const float* pp, unsigned pp_height, unsigned pp_width  )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= domain_width || iy >= lookup_prop  ) return;

    qprop qp(pp, pp_width, pp_height ); 

    float x = domain[ix] ; 
    float y = qp.interpolate( iy, x ); 
    unsigned index = iy * domain_width + ix ;

    if( iy == 0 )
    printf("//_QProp_lookup ix %3d x %10.4f  iy %d  y %10.4f pp_width %3d pp_height %3d \n", ix, x, iy, y, pp_width, pp_height ); 

    lookup[index] = y ; 
}

extern "C" void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    float* lookup, 
    const float* domain, 
    unsigned lookup_prop, 
    unsigned domain_width, 
    const float* pp, 
    unsigned pp_height, 
    unsigned pp_width  )
{
    _QProp_lookup<<<numBlocks,threadsPerBlock>>>( lookup, domain, lookup_prop, domain_width, pp, pp_height, pp_width ) ;
} 

