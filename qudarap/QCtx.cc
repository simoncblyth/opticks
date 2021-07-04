
#include "PLOG.hh"
#include "SSys.hh"
#include "scuda.h"

#include "NPY.hpp"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "GScintillatorLib.hh"

#include "QUDA_CHECK.h"
#include "QU.hh"

#include "qctx.h"

#include "QRng.hh"
#include "QTex.hh"
#include "QScint.hh"
#include "QBnd.hh"
#include "QCtx.hh"

const plog::Severity QCtx::LEVEL = PLOG::EnvLevel("QCtx", "INFO"); 
const QCtx* QCtx::INSTANCE = nullptr ; 
const QCtx* QCtx::Get(){ return INSTANCE ; }

void QCtx::Init(const GGeo* ggeo)
{
    GScintillatorLib* slib = ggeo->getScintillatorLib(); 
    slib->dump();

    GBndLib* blib = ggeo->getBndLib(); 
    blib->createDynamicBuffers();  // hmm perhaps this is done already on loading now ?
    blib->dump(); 

    // on heap, to avoid dtors

    QRng* qrng = new QRng ;  // loads and uploads curandState 
    LOG(LEVEL) << qrng->desc(); 


    QScint* qscint = nullptr ;  

    const char* qctx_icdf_path = SSys::getenvvar("QCTX_ICDF_PATH", nullptr ); 
    NPY<double>* icdf = qctx_icdf_path == nullptr ? nullptr : NPY<double>::load(qctx_icdf_path) ; 
    if( icdf == nullptr )
    {
        LOG(LEVEL) << " booting QScint from standard GScintillatorLib " ; 
        qscint = new QScint(slib); 
    }
    else
    {
        LOG(LEVEL) 
            << " booting QScint from non-standard icdf " << icdf->getShapeString() 
            << " loaded from QCTX_ICDF_PATH " << qctx_icdf_path 
            ; 
        qscint = new QScint(icdf); 
    }
    LOG(LEVEL) << qscint->desc(); 

    QBnd* qbnd = new QBnd(blib); 
    LOG(LEVEL) << qbnd->desc(); 
}


QCtx::QCtx()
    :
    rng(QRng::Get()),
    scint(QScint::Get()),
    bnd(QBnd::Get()),
    ctx(new qctx),
    d_ctx(nullptr)
{
    INSTANCE = this ; 
    init(); 
}

/**
QCtx::init
------------

Collect device side refs/handles into qctx(ctx) and upload it to d_ctx

**/
void QCtx::init()
{
    LOG(LEVEL) 
        << " rng " << rng 
        << " scint " << scint
        << " bnd " << bnd
        << " ctx " << ctx 
        << " d_ctx " << d_ctx 
        ;  

    if(rng)
    {
        LOG(LEVEL) << " rng " << rng->desc() ; 
        ctx->r = rng->d_rng_states ; 
    } 
    if(scint)
    {
        LOG(LEVEL) << " scint " << scint->desc() ; 
        ctx->scint_tex = scint->tex->texObj ; 
        ctx->scint_meta = scint->tex->d_meta ; 
    } 
    if(bnd)
    {
        LOG(LEVEL) << " bnd " << bnd->desc() ; 
        ctx->boundary_tex = bnd->tex->texObj ; 
        ctx->boundary_meta = bnd->tex->d_meta ; 
    } 

    d_ctx = QU::UploadArray<qctx>(ctx, 1 );  

    LOG(LEVEL) << desc() ; 
}


std::string QCtx::desc() const
{
    std::stringstream ss ; 
    ss << "QCtx"
       << " ctx->r " << ctx->r 
       << " ctx->scint_tex " << ctx->scint_tex 
       << " ctx->scint_meta " << ctx->scint_meta
       << " ctx->boundary_tex " << ctx->boundary_tex 
       << " ctx->boundary_meta " << ctx->boundary_meta
       << " d_ctx " << d_ctx 
       ; 
    std::string s = ss.str(); 
    return s ; 
}


extern "C" void QCtx_generate_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* wavelength, unsigned num_wavelength, unsigned hd_factor ); 
extern "C" void QCtx_generate_photon(    dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, quad4* photon    , unsigned num_photon ); 


void QCtx::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

void QCtx::generate( float* wavelength, unsigned num_wavelength, unsigned hd_factor )
{
    LOG(LEVEL) << "[" ; 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_wavelength, 1 ); 

    size_t size = num_wavelength*sizeof(float) ; 

    float* d_wavelength ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_wavelength ), size )); 

    QCtx_generate_wavelength(numBlocks, threadsPerBlock, d_ctx, d_wavelength, num_wavelength, hd_factor );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( wavelength ), d_wavelength, size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_wavelength) ); 
    LOG(LEVEL) << "]" ; 
}

void QCtx::dump( float* wavelength, unsigned num_wavelength, unsigned edgeitems )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_wavelength ; i++)
    {
        if( i < edgeitems || i > num_wavelength - edgeitems)
        {
            std::cout 
                << std::setw(10) << i 
                << std::setw(10) << std::fixed << std::setprecision(3) << wavelength[i] 
                << std::endl 
                ; 
        }
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

    QCtx_generate_photon(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( photon ), d_photon, sizeof(quad4)*num_photon, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_photon) ); 
    LOG(LEVEL) << "]" ; 
}

void QCtx::dump( quad4* photon, unsigned num_photon, unsigned edgeitems )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_photon ; i++)
    {
        if( i < edgeitems || i > num_photon - edgeitems)
        {
            const quad4& p = photon[i] ;  
            std::cout 
                << std::setw(10) << i 
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
}

