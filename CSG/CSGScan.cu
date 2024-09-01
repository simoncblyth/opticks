/**
CSGScan.cu
===========


**/

#include "CSGParams.h"


__global__ void _CSGScan_intersect( CSGParams* d )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= d->num ) return;

    d->intersect(ix); 
}

extern void CSGScan_intersect(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    CSGParams* d )
{
    _CSGScan_intersect<<<numBlocks,threadsPerBlock>>>( d ) ;
}

