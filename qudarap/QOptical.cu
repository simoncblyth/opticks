
#include <stdio.h>
#include "scuda.h"
#include "squad.h"


__global__ void _QOptical_check( quad* optical, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = iy * width + ix ;
    if (ix >= width | iy >= height ) return;

    uint4& u = optical[iy].u ; 

    printf("//_QOptical_check ix %3d iy %3d index %3d  optical[iy] (%3d %3d %3d %3d)   \n", ix, iy, index, u.x, u.y, u.z, u.w ); 
}

extern "C" void QOptical_check(dim3 numBlocks, dim3 threadsPerBlock, quad* optical, unsigned width, unsigned height ) 
{
    _QOptical_check<<<numBlocks,threadsPerBlock>>>( optical, width, height );
} 


