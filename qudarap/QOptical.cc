
#include <sstream>

#include "scuda.h"
#include "squad.h"
#include "NP.hh"

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
#include <cuda_runtime.h>
#include "QUDA_CHECK.h"
#include "QU.hh"
#include "QBuf.hh"
#include "SLOG.hh"
#endif

#include "QOptical.hh"

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
const plog::Severity QOptical::LEVEL = SLOG::EnvLevel("QOptical", "DEBUG");
#endif

const QOptical* QOptical::INSTANCE = nullptr ; 
const QOptical* QOptical::Get(){ return INSTANCE ; }

QOptical::QOptical(const NP* optical_)
    :
    optical(optical_),
    buf(nullptr),
    d_optical(nullptr)
{
    init();
} 

void QOptical::init()
{
    INSTANCE = this ; 

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)

    NP* a = const_cast<NP*>(optical); 
    d_optical = a->values<quad>(); 
    assert( d_optical ); 

#else
    buf = QBuf<unsigned>::Upload(optical) ; 
    d_optical = (quad*)buf->d ; 
#endif
}


std::string QOptical::desc() const
{
    std::stringstream ss ; 
    ss << "QOptical"
       << " optical " << ( optical ? optical->desc() : "-" )
       ;   
    std::string s = ss.str(); 
    return s ; 
}


#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
void QOptical::check() const {}
#else

// from QOptical.cu
extern "C" void QOptical_check(dim3 numBlocks, dim3 threadsPerBlock, quad* optical, unsigned width, unsigned height ); 

void QOptical::check() const 
{
    unsigned height = optical->shape[0] ; 
    unsigned width = 1 ; 

    LOG(LEVEL) 
         << "[" << " height " << height << " width " << width 
         ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch( numBlocks, threadsPerBlock, width, height ); 

    QOptical_check(numBlocks, threadsPerBlock, d_optical, width, height );  

    LOG(LEVEL) << "]" ; 
}
#endif

