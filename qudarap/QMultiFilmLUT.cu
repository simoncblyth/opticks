

#include "curand_kernel.h"
#include "scuda.h"
#include "squad.h"

//#include "qscint.h"
#include "stdio.h"

/**

**/

__global__ void _QMultiFilmLUT_check(unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
   // printf("hello world!");
    if( ix % 10 == 0 )   printf( "//_QMultiFilmLUT_check  ix %d iy %d width %d height %d \n", ix, iy, width, height ); 
}

__global__ void _QMultiFilmLUT_lookup(cudaTextureObject_t tex, quad4* meta, float4* lookup, unsigned num_lookup, unsigned width, unsigned height)
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    int index = iy * width + ix;
    unsigned nx = meta->q0.u.x ; 
    unsigned ny = meta->q0.u.y ; 

    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;

    float4 v = tex2D<float4>( tex, x, y );     

    //printf( "//_QMultiFilmLUT_lookup  ix %d iy %d width %d height %d index %d nx %d ny %d x %10.4f y %10.4f v.x %10.4f v.y %10.4f v.z %10.4f v.w %10.4f  \n", ix, iy, width, height, index, nx, ny, x, y, v.x, v.y, v.z, v.w ); 
    lookup[ index ] = v ;  
}

extern "C" void QMultiFilmLUT_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t tex, quad4* meta, float4* lookup, unsigned num_lookup, unsigned width, unsigned height) 
{
    printf("//QMultiFilmLUT_lookup num_lookup %d width %d height %d \n", num_lookup, width, height);  
    _QMultiFilmLUT_lookup<<<numBlocks,threadsPerBlock>>>( tex, meta, lookup, num_lookup, width, height );
} 

extern "C" void QMultiFilmLUT_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height ) 
{
    printf("//QMultiFilmLUT_check width %d height %d threadsPerBlock.x %d threadsPerBlock.y %d  numBlocks.x %d numBlocks.y %d \n", width, height,threadsPerBlock.x, threadsPerBlock.y , numBlocks.x, numBlocks.y);  

    _QMultiFilmLUT_check<<<numBlocks,threadsPerBlock>>>( width, height  );
} 




