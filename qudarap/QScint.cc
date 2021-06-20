
#include "PLOG.hh"
#include "NPY.hpp"
#include "GScintillatorLib.hh"

#include "QUDA_CHECK.h"
#include "QRng.hh"
#include "QScint.hh"
#include "QTex2D.hh"

const plog::Severity QScint::LEVEL = PLOG::EnvLevel("QScint", "INFO"); 

QScint::QScint(const GScintillatorLib* lib_)
    :
    lib(lib_),
    buf(lib->getBuffer()),
    ni(buf->getShape(0)),
    nj(buf->getShape(1)),
    nk(buf->getShape(2)),
    tex(new QTex2D<float>(nj, 1, buf->getValues())),
    rng(QRng::Get())
{
    init(); 
}

void QScint::init()
{
    LOG(LEVEL) 
        << " buf " << ( buf ? buf->getShapeString() : "-" ) 
        << " tex " << tex 
        ;  

    assert( ni == 1 ); 
    assert( nj == 4096 ); 
    assert( nk == 1 ); 
}


extern "C" void QScint_generate_kernel(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength ); 


void QScint::generate( float* wavelength, unsigned num_wavelength )
{
    float* d_wavelength ;  

    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_wavelength ), num_wavelength*sizeof(float) )); 

    unsigned width = num_wavelength  ; 
    unsigned height = 1 ; 

    dim3 threadsPerBlock(512, 1);

    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    QScint_generate_kernel(numBlocks, threadsPerBlock, rng->d_rng_states, tex->texObj, d_wavelength, num_wavelength );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( wavelength ), d_wavelength, sizeof(float)*num_wavelength, cudaMemcpyDeviceToHost )); 

    QUDA_CHECK( cudaFree(d_wavelength) ); 

    LOG(LEVEL) << "]" ; 
}

void QScint::dump( float* wavelength, unsigned num_wavelength )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_wavelength ; i++)
    {
        std::cout 
            << std::setw(3) << i 
            << std::setw(10) << std::fixed << std::setprecision(3) << wavelength[i] 
            << std::endl 
            ; 
    }
}


