#include "QTexMaker.hh"

#include "scuda.h"
#include "NP.hh"
#include "QTex.hh"
#include "PLOG.hh"

const plog::Severity QTexMaker::LEVEL = PLOG::EnvLevel("QTexMaker", "DEBUG"); 


QTex<float4>* QTexMaker::Make2d_f4( const NP* icdf, char filterMode )  // static 
{
    unsigned ndim = icdf->shape.size(); 
    unsigned hd_factor = icdf->get_meta<unsigned>("hd_factor", 0) ; 

    if( filterMode == 'P' ) LOG(fatal) << " filtermode 'P' without interpolation is in use : appropriate for basic tex machinery tests only " ; 

    LOG(LEVEL)
        << "["  
        << " icdf " << icdf->sstr()
        << " ndim " << ndim 
        << " hd_factor " << hd_factor 
        << " filterMode " << filterMode 
        ;

    assert( ndim == 3 && icdf->shape[ndim-1] == 4 ); 

    QTex<float4>* tex = QTexMaker::Make2d_f4_(icdf, filterMode ); 
    tex->setHDFactor(hd_factor); 
    tex->uploadMeta(); 

    LOG(LEVEL) << "]" ; 

    return tex ; 
}




QTex<float4>* QTexMaker::Make2d_f4_( const NP* a, char filterMode )  // static 
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
    tex->setOrigin(a); 

    return tex ; 
}

