#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

#include "PLOG.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "scuda.h"

#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QCerenkov.hh"
#include "QTex.hh"
#include "QTexMaker.hh"
#include "QTexLookup.hh"

#include "QCK.hh"


const plog::Severity QCerenkov::LEVEL = PLOG::EnvLevel("QCerenkov", "DEBUG"); 

const QCerenkov* QCerenkov::INSTANCE = nullptr ; 
const QCerenkov* QCerenkov::Get(){ return INSTANCE ;  }


const char* QCerenkov::DEFAULT_FOLD = "$TMP/QCerenkovIntegralTest/test_makeICDF_SplitBin" ; 

NP* QCerenkov::Load(const char* fold, const char* name)  // static
{
    int create_dirs = 0 ;  // 0:nop
    const char* path = SPath::Resolve(fold, name, create_dirs);  
    NP* a = NP::Load(path); 

    if( a == nullptr )
    {
        LOG(fatal)
            << " failed to load array from path " << path 
            ; 
    }
    return a ; 
}


/**
QCerenkov::MakeTex
--------------------

**/

QTex<float4>* QCerenkov::MakeTex(const NP* icdf, char filterMode, bool normalizedCoords ) // static
{
    LOG(info) << " icdf " << icdf << " icdf.sstr " << icdf->sstr() << " filterMode " << filterMode ; 
    QTex<float4>* tex = QTexMaker::Make2d_f4(icdf, filterMode, normalizedCoords ); 
    LOG(info) << " tex " << tex ; 
    return tex ; 
}


/**
QCerenkov::QCerenkov
----------------------

Currently relies on icdf created by QCerenkovIntegralTest

TODO: formalize icdf creation into the pre-cache geometry conversion and icdf location into geocache  

**/

QCerenkov::QCerenkov(const char* fold_ )
    :
    fold( fold_ ? fold_ : DEFAULT_FOLD ),
    icdf_( nullptr ),
    icdf( nullptr ),
    filterMode('P'),    // CAUTION: P: is Point mode with interpolation disabled
    normalizedCoords(true), 
    tex(nullptr),
    look(nullptr)
{
    init(); 
}

void QCerenkov::init()
{
    INSTANCE = this ; 
    icdf_ = Load(fold, "icdf.npy"); 
    if( icdf_ == nullptr )
    {
        LOG(fatal) << "failed to load icdf.npy from fold " << fold ; 
        return ;  
    }

    LOG(info) 
        << " icdf_ " << icdf_ 
        << " icdf_.sstr " << icdf_->sstr() 
        << " icdf_.meta " << icdf_->meta 
        ; 

    icdf = icdf_->ebyte == 4 ? icdf_ : NP::MakeNarrow(icdf_) ; 

    LOG(info) 
        << " icdf " << icdf 
        << " icdf.sstr " << icdf->sstr()
        << " icdf.meta " << icdf->meta 
        ; 

    tex = MakeTex(icdf, filterMode, normalizedCoords ) ; 

    LOG(info) << " tex " << tex ; 

    look = new QTexLookup<float4>( tex ) ; 

    LOG(info) << " look " << look ; 
}


NP* QCerenkov::lookup() 
{
    return look ? look->lookup() : nullptr ; 
}


std::string QCerenkov::desc() const
{
    std::stringstream ss ; 
    ss << "QCerenkov"
       << " fold " << fold
       << " icdf_ " << ( icdf_ ? icdf_->sstr() : "-" )
       << " icdf " << ( icdf ? icdf->sstr() : "-" )
       << " tex " << tex 
       ; 

    std::string s = ss.str(); 
    return s ; 
}




extern "C" void QCerenkov_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height  ); 

template <typename T>
extern void QCerenkov_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, T* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 


void QCerenkov::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
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

void QCerenkov::check()
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
    QCerenkov_check(numBlocks, threadsPerBlock, width, height );  

    cudaDeviceSynchronize();
}


void QCerenkov::dump( float* lookup, unsigned num_lookup, unsigned edgeitems  )
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

