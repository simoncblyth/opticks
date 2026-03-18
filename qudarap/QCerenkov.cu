/**
QCerenkov.cu
=============

Note:

* used only(?) from tests/QCerenkovTest.cc
* operates at low level with texture arguments, no qcerenkov.h dependency

**/


#include "scuda.h"
#include "squad.h"
#include "stdio.h"

__global__ void _QCerenkov_check(unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    printf( "//_QCerenkov_check  ix %d iy %d width %d height %d \n", ix, iy, width, height );
}

__global__ void _QCerenkov_lookup(cudaTextureObject_t tex, quad4* d_meta, float* lookup, unsigned num_lookup, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    unsigned index = iy * width + ix ;

    unsigned nx = d_meta ? d_meta->q0.u.x : 0 ;
    unsigned ny = d_meta ? d_meta->q0.u.y : 0 ;

    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;

    float v = tex2D<float>( tex, x, y );

    if( ix % 100 == 0 ) printf( "//_QCerenkov_lookup  ix %d iy %d width %d height %d index %d nx %d ny %d x %10.4f y %10.4f v %10.4f \n", ix, iy, width, height, index, nx, ny, x, y, v );
    lookup[index] = v ;
}

extern "C" void QCerenkov_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t tex, quad4* meta, float* lookup, unsigned num_lookup, unsigned width, unsigned height )
{
    printf("//QCerenkov_lookup num_lookup %d width %d height %d \n", num_lookup, width, height );
    _QCerenkov_lookup<<<numBlocks,threadsPerBlock>>>( tex, meta, lookup, num_lookup, width, height  );
}

extern "C" void QCerenkov_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height )
{
    printf("//QCerenkov_check width %d height %d \n", width, height );
    _QCerenkov_check<<<numBlocks,threadsPerBlock>>>( width, height  );
}


