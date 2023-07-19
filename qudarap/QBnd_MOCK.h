#pragma once

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#include "stexture.h"

MockTextureManager* MockTextureManager::INSTANCE = nullptr ; 

extern "C" void QBnd_lookup_0_MOCK(
    cudaTextureObject_t tex, 
    quad4* meta, 
    quad* lookup, 
    int num_lookup, 
    int width, 
    int height 
    )
{
    unsigned nx = meta->q0.u.x  ; 
    unsigned ny = meta->q0.u.y  ; 

    for(int iy=0 ; iy < height ; iy++)
    for(int ix=0 ; ix < width ; ix++)
    {
        int index = iy * width + ix ;
        float x = (float(ix)+0.5f)/float(nx) ;
        float y = (float(iy)+0.5f)/float(ny) ;
        quad q ; 
        q.f = tex2D<float4>( tex, x, y );     
#ifdef DEBUG
        // debug launch config by returning coordinates 
        printf(" ix %d iy %d index %d nx %d ny %d x %10.3f y %10.3f \n", ix, iy, index, nx, ny, x, y ); 
        q.u.x = ix ; 
        q.u.y = iy ; 
        q.u.z = index ; 
        q.u.w = nx ; 
#endif
        lookup[index] = q ; 
    }
}

#endif


