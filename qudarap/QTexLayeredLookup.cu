#include "scuda.h"
#include "squad.h"
#include "stdio.h"


template <typename T>
__global__ void _QTexLayeredLookup_lookup(cudaTextureObject_t tex, quad4* d_meta, T* lookup, unsigned layer )
{
    unsigned ix = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned nx = d_meta ? d_meta->q0.u.x : 0 ;
    unsigned ny = d_meta ? d_meta->q0.u.y : 0 ;
    unsigned nl = d_meta ? d_meta->q0.u.z : 0 ;
    if (ix >= nx || iy >= ny || layer >= nl ) return;

    unsigned index = layer * nx * ny + iy * nx + ix ;

    float x = (float(ix)+0.5f)/float(nx) ;
    float y = (float(iy)+0.5f)/float(ny) ;

    T value = tex2DLayered<T>(tex, x, y, layer);

    //bool dump = ix % 100 == 0 ;
    bool dump = true ;
    //bool dump = index > 90 ;
    // unclear how to dump value while templated

    if(dump) printf( "//_QTexLayeredLookup_lookup  ix %d iy %d layer %d index %d nx %d ny %d nl %d x %10.4f y %10.4f \n", ix, iy, layer, index, nx, ny, nl, x, y );

    lookup[index] = value ;
}


template <typename T>
extern void QTexLayeredLookup_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t tex, quad4* meta, T* lookup, unsigned layer  )
{
    printf("//QTexLayeredLookup_lookup layer %d \n", layer);
    _QTexLayeredLookup_lookup<T><<<numBlocks,threadsPerBlock>>>(tex, meta, lookup, layer );
}

template void QTexLayeredLookup_lookup(dim3, dim3, cudaTextureObject_t, quad4*, float4* , unsigned );
template void QTexLayeredLookup_lookup(dim3, dim3, cudaTextureObject_t, quad4*, float*  , unsigned );

