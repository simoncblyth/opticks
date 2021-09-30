#include "QTexMaker.hh"

#include "scuda.h"
#include "NP.hh"
#include "QTex.hh"

QTex<float4>* QTexMaker::Make2d_f4( const NP* a, char filterMode )  // static 
{
    assert( a->ebyte == 4 && "need to narrow double precision arrays first ");  
    unsigned ndim = a->shape.size(); 
    assert( ndim == 3 );      

    unsigned ni = a->shape[0] ; 
    unsigned nj = a->shape[1] ; 
    unsigned nk = a->shape[2] ; assert( nk == 4 ); 

    size_t height = ni ; 
    size_t  width = nj ; 
    const void* src = (const void*)a->bytes(); 

    // note that from the point of view of array content, saying (height, width) 
    // is a more appropriate ordering than the usual contrary convention  
    
    QTex<float4>* tex = new QTex<float4>( width, height, src, filterMode  );
    return tex ; 
}

