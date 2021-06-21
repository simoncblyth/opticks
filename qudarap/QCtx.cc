
#include "PLOG.hh"
#include "scuda.h"

#include "NPY.hpp"
#include "GScintillatorLib.hh"

#include "QUDA_CHECK.h"
#include "QRng.hh"
#include "QCtx.hh"
#include "QTex.hh"


const plog::Severity QCtx::LEVEL = PLOG::EnvLevel("QCtx", "INFO"); 

QCtx::QCtx(const GScintillatorLib* lib_)
    :
    lib(lib_),
    buf(lib->getBuffer()),
    ni(buf->getShape(0)),
    nj(buf->getShape(1)),
    nk(buf->getShape(2)),
    tex(new QTex<float>(nj, 1, buf->getValues())),
    rng(QRng::Get())
{
    init(); 
}

void QCtx::init()
{
    LOG(LEVEL) 
        << " buf " << ( buf ? buf->getShapeString() : "-" ) 
        << " tex " << tex 
        ;  

    assert( ni == 1 ); 
    assert( nj == 4096 ); 
    assert( nk == 1 ); 
}


extern "C" void QCtx_generate_wavelength(dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, float* wavelength, unsigned num_wavelength ); 
extern "C" void QCtx_generate_photon(    dim3 numBlocks, dim3 threadsPerBlock, curandState* rng_states, cudaTextureObject_t texObj, quad4* photon    , unsigned num_photon ); 


void QCtx::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

void QCtx::generate( float* wavelength, unsigned num_wavelength )
{
    LOG(LEVEL) << "[" ; 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_wavelength, 1 ); 

    float* d_wavelength ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_wavelength ), num_wavelength*sizeof(float) )); 

    QCtx_generate_wavelength(numBlocks, threadsPerBlock, rng->d_rng_states, tex->texObj, d_wavelength, num_wavelength );  
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( wavelength ), d_wavelength, sizeof(float)*num_wavelength, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_wavelength) ); 

    LOG(LEVEL) << "]" ; 
}

void QCtx::dump( float* wavelength, unsigned num_wavelength )
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

void QCtx::generate( quad4* photon, unsigned num_photon )
{
    LOG(LEVEL) << "[" ; 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_photon, 1 ); 

    quad4* d_photon ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_photon ), num_photon*sizeof(quad4) )); 

    QCtx_generate_photon(numBlocks, threadsPerBlock, rng->d_rng_states, tex->texObj, d_photon, num_photon );  
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( photon ), d_photon, sizeof(quad4)*num_photon, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_photon) ); 

    LOG(LEVEL) << "]" ; 
}

void QCtx::dump( quad4* photon, unsigned num_photon )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_photon ; i++)
    {
        const quad4& p = photon[i] ;  
        std::cout 
            << std::setw(3) << i 
            << " q1.f.xyz " 
            << std::setw(10) << std::fixed << std::setprecision(3) << p.q1.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << p.q1.f.y
            << std::setw(10) << std::fixed << std::setprecision(3) << p.q1.f.z  
            << " q2.f.xyz " 
            << std::setw(10) << std::fixed << std::setprecision(3) << p.q2.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << p.q2.f.y
            << std::setw(10) << std::fixed << std::setprecision(3) << p.q2.f.z  
            << std::endl 
            ; 
    }
}


