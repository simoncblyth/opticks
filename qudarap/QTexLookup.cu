#include "scuda.h"
#include "squad.h"
#include "stdio.h"


template <typename T>
__global__ void _QTexLookup_lookup(cudaTextureObject_t tex, quad4* d_meta, T* lookup )
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned nx = d_meta ? d_meta->q0.u.x : 0 ;
    unsigned ny = d_meta ? d_meta->q0.u.y : 0 ;

    if (ix >= nx || iy >= ny ) return;

    unsigned index = iy * nx + ix ;
    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;   // scaling like this means need normalizeCoordinates:true to match origin and lookup

    T v = tex2D<T>( tex, x, y );

    bool dump = ix % 100 == 0 ;
    //bool dump = true ;

    if( dump) printf( "//_QTexLookup_lookup  ix %d iy %d index %d nx %d ny %d x %10.4f y %10.4f \n", ix, iy, index, nx, ny, x, y );


    lookup[index] = v ;
}


template <typename T>
extern void QTexLookup_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t tex, quad4* meta, T* lookup )
{
    printf("//QTexLookup_lookup \n" );
    _QTexLookup_lookup<T><<<numBlocks,threadsPerBlock>>>(tex, meta, lookup );
}

template void QTexLookup_lookup(dim3, dim3, cudaTextureObject_t, quad4*, float4* ) ;
template void QTexLookup_lookup(dim3, dim3, cudaTextureObject_t, quad4*, float*  );

