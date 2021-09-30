
#include "PLOG.hh"
#include "SSys.hh"
#include "scuda.h"

#include <sstream>

#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QRng.hh"
#include "QScint.hh"
#include "QTex.hh"

const plog::Severity QScint::LEVEL = PLOG::EnvLevel("QScint", "INFO"); 

const QScint* QScint::INSTANCE = nullptr ; 
const QScint* QScint::Get(){ return INSTANCE ;  }


QScint::QScint(const NP* icdf, unsigned hd_factor )
    :
    dsrc(icdf->ebyte == 8 ? icdf : nullptr),
    src( icdf->ebyte == 4 ? icdf : NP::MakeNarrow(dsrc) ), 
    tex(MakeScintTex(src, hd_factor))
{
    INSTANCE = this ; 
}

std::string QScint::desc() const
{
    std::stringstream ss ; 
    ss << "QScint"
       << " dsrc " << ( dsrc ? dsrc->desc() : "-" )
       << " src " << ( src ? src->desc() : "-" )
       << " tex " << ( tex ? tex->desc() : "-" )
       << " tex " << tex 
       ; 

    std::string s = ss.str(); 
    return s ; 
}

/**
QScint::MakeScintTex
-----------------------

TODO: move the hd_factor into payload instead of items for easier extension to 2d 

**/

QTex<float>* QScint::MakeScintTex(const NP* src, unsigned hd_factor )  // static 
{
    assert( src->has_shape(1,4096,1) ||  src->has_shape(3,4096,1) ); 
    assert( src->uifc == 'f' ); 
    assert( src->ebyte == 4 );    // expecting float src array, possible narrowed from double dsrc array  

    unsigned ni = src->shape[0]; 
    unsigned nj = src->shape[1]; 
    unsigned nk = src->shape[2]; 

    assert( ni == 1 || ni == 3); 
    assert( nj == 4096 ); 
    assert( nk == 1 ); 

    unsigned ny = ni ; // height  
    unsigned nx = nj ; // width 
  

    bool qscint_disable_interpolation = SSys::getenvbool("QSCINT_DISABLE_INTERPOLATION"); 
    char filterMode = qscint_disable_interpolation ? 'P' : 'L' ; 

    if(qscint_disable_interpolation)
        LOG(fatal) << "QSCINT_DISABLE_INTERPOLATION active using filterMode " << filterMode 
        ; 

    QTex<float>* tx = new QTex<float>(nx, ny, src->cvalues<float>(), filterMode ) ; 

    tx->setHDFactor(hd_factor); 
    tx->uploadMeta(); 

    LOG(LEVEL)
        << " src " << src->desc()
        << " nx (width) " << nx
        << " ny (height) " << ny
        << " tx.HDFactor " << tx->getHDFactor() 
        << " tx.filterMode " << tx->getFilterMode()
        ;

    return tx ; 
}

extern "C" void QScint_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height  ); 
extern "C" void QScint_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, float* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 

void QScint::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 

    LOG(LEVEL) 
        << " width " << std::setw(7) << width 
        << " height " << std::setw(7) << height 
        << " width*height " << std::setw(7) << width*height 
        << " threadsPerBlock"
        << "(" 
        << std::setw(3) << threadsPerBlock.x << " " 
        << std::setw(3) << threadsPerBlock.y << " " 
        << std::setw(3) << threadsPerBlock.z << " "
        << ")" 
        << " numBlocks "
        << "(" 
        << std::setw(3) << numBlocks.x << " " 
        << std::setw(3) << numBlocks.y << " " 
        << std::setw(3) << numBlocks.z << " "
        << ")" 
        ;
}

void QScint::check()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        ;

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 
    QScint_check(numBlocks, threadsPerBlock, width, height );  

    cudaDeviceSynchronize();
}


NP* QScint::lookup()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        << " lookup " << num_lookup
        ;

    NP* out = NP::Make<float>(height, width ); 

    float* out_ = out->values<float>(); 
    lookup( out_ , num_lookup, width, height ); 

    return out ; 
}

void QScint::lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height  )
{
    LOG(LEVEL) << "[" ; 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 
    
    size_t size = width*height*sizeof(float) ; 
  
    LOG(LEVEL) 
        << " num_lookup " << num_lookup
        << " width " << width 
        << " height " << height
        << " size " << size 
        << " tex->texObj " << tex->texObj
        << " tex->meta " << tex->meta
        << " tex->d_meta " << tex->d_meta
        ; 

    float* d_lookup = nullptr ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size )); 

    QScint_lookup(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lookup ), d_lookup, size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_lookup) ); 

    cudaDeviceSynchronize();

    LOG(LEVEL) << "]" ; 
}

void QScint::dump( float* lookup, unsigned num_lookup, unsigned edgeitems  )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_lookup ; i++)
    {
        if( i < edgeitems || i > num_lookup - edgeitems )
        std::cout 
            << std::setw(6) << i 
            << std::setw(10) << std::fixed << std::setprecision(3) << lookup[i] 
            << std::endl 
            ; 
    }
}

