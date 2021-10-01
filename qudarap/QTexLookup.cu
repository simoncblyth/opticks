#include "scuda.h"
#include "squad.h"
#include "stdio.h"


template <typename T>
__global__ void _QTexLookup_lookup(cudaTextureObject_t tex, quad4* d_meta, T* lookup, unsigned num_lookup, unsigned width, unsigned height )
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    
    unsigned index = iy * width + ix ;

    unsigned nx = d_meta ? d_meta->q0.u.x : 0 ;
    unsigned ny = d_meta ? d_meta->q0.u.y : 0 ; 

    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;

    T v = tex2D<T>( tex, x, y );

    if( ix % 100 == 0 ) printf( "//_QTexLookup_lookup  ix %d iy %d width %d height %d index %d nx %d ny %d x %10.4f y %10.4f \n", ix, iy, width, height, index, nx, ny, x, y );

    lookup[index] = v ;
}   


template <typename T>
extern void QTexLookup_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t tex, quad4* meta, T* lookup, unsigned num_lookup, unsigned width, unsigned height  )
{
    printf("//QTexLookup_lookup num_lookup %d height %d width %d \n", num_lookup, height, width );  
    _QTexLookup_lookup<T><<<numBlocks,threadsPerBlock>>>(tex, meta, lookup, num_lookup, width, height );
} 

template void QTexLookup_lookup(dim3, dim3, cudaTextureObject_t, quad4*, float4* , unsigned, unsigned, unsigned ); 
template void QTexLookup_lookup(dim3, dim3, cudaTextureObject_t, quad4*, float*  , unsigned, unsigned, unsigned ); 

