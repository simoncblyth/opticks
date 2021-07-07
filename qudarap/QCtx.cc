
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
    bool qctx_dump = SSys::getenvbool("QCTX_DUMP"); 

    GScintillatorLib* slib = ggeo->getScintillatorLib(); 
    if(qctx_dump) slib->dump();

    GBndLib* blib = ggeo->getBndLib(); 
    blib->createDynamicBuffers();  // hmm perhaps this is done already on loading now ?
    if(qctx_dump) blib->dump(); 

    // on heap, to avoid dtors

    QRng* qrng = new QRng ;  // loads and uploads curandState 
    LOG(LEVEL) << qrng->desc(); 

    QScint* qscint = MakeScint(slib); 
    LOG(LEVEL) << qscint->desc(); 

    QBnd* qbnd = new QBnd(blib); 
    LOG(LEVEL) << qbnd->desc(); 
}

QScint* QCtx::MakeScint(const GScintillatorLib* slib)
{
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
    return qscint ; 
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

    unsigned hd_factor = scint->tex->getHDFactor() ;  // HMM: perhaps get this from ctx rather than occupying an argument slot  
    LOG(LEVEL) 
        << " hd_factor " << hd_factor  
        ;

    if(rng)
    {
        LOG(LEVEL) << " rng " << rng->desc() ; 
        ctx->r = rng->d_rng_states ; 
    } 
    if(scint)
    {
        LOG(LEVEL) << " scint.desc " << scint->desc() ; 
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

char QCtx::getScintTexFilterMode() const 
{
    return scint->tex->getFilterMode() ; 
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


void QCtx::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}



template<typename T>
T* QCtx::device_alloc( unsigned num_items )
{
    size_t size = num_items*sizeof(T) ; 
    T* d ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d ), size )); 
    return d ; 
}

template<typename T>
void QCtx::device_free( T* d)
{
    QUDA_CHECK( cudaFree(d) ); 
}


template<typename T>
void QCtx::copy_device_to_host( T* h, T* d,  unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d) ); 
}

template<typename T>
void QCtx::copy_host_to_device( T* d, T* h, unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d ), h , size, cudaMemcpyHostToDevice )); 
}



/**
QCtx::generate_scint
----------------------

Setting envvar QCTX_DISABLE_HD disables multiresolution handling
and causes the returned hd_factor to be zero rather then 
the typical values of 10 or 20 which depend on the buffer creation.

**/

extern "C" void QCtx_generate_scint_wavelength(   dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* wavelength, unsigned num_wavelength, unsigned hd_factor ); 
void QCtx::generate_scint( float* wavelength, unsigned num_wavelength, unsigned& hd_factor )
{
    bool qctx_disable_hd = SSys::getenvbool("QCTX_DISABLE_HD"); 
    hd_factor = qctx_disable_hd ? 0u : scint->tex->getHDFactor() ; 
    // HMM: perhaps get this from ctx rather than occupying an argument slot  
    LOG(LEVEL) << "[" << " qctx_disable_hd " << qctx_disable_hd << " hd_factor " << hd_factor ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_wavelength, 1 ); 

    float* d_wavelength = device_alloc<float>(num_wavelength); 

    QCtx_generate_scint_wavelength(numBlocks, threadsPerBlock, d_ctx, d_wavelength, num_wavelength, hd_factor );  

    copy_device_to_host<float>( wavelength, d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 
}

/**
QCtx::generate_cerenkov
-------------------------

**/

extern "C" void QCtx_generate_cerenkov_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* wavelength, unsigned num_wavelength ); 
void QCtx::generate_cerenkov( float* wavelength, unsigned num_wavelength )
{
    LOG(LEVEL) << "[" ; 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_wavelength, 1 ); 

    float* d_wavelength = device_alloc<float>(num_wavelength); 

    QCtx_generate_cerenkov_wavelength(numBlocks, threadsPerBlock, d_ctx, d_wavelength, num_wavelength );  

    copy_device_to_host<float>( wavelength, d_wavelength, num_wavelength ); 

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



extern "C" void QCtx_generate_photon(    dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, quad4* photon , unsigned num_photon ); 
void QCtx::generate( quad4* photon, unsigned num_photon )
{
    LOG(LEVEL) << "[" ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_photon, 1 ); 

    quad4* d_photon = device_alloc<quad4>(num_photon) ; 

    QCtx_generate_photon(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon );  

    copy_device_to_host<quad4>( photon, d_photon, num_photon ); 

    LOG(LEVEL) << "]" ; 
}


extern "C" void QCtx_boundary_lookup_all(    dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, quad* lookup  , unsigned width, unsigned height ); 
void QCtx::boundary_lookup_all(quad* lookup, unsigned width, unsigned height )
{
    LOG(LEVEL) << "[" ; 
    assert( bnd ); 
    assert( width <= getBoundaryTexWidth()  ); 
    assert( height <= getBoundaryTexHeight()  ); 

    unsigned num_lookup = width*height ; 
    LOG(LEVEL) 
        << " width " << width 
        << " height " << height 
        << " num_lookup " << num_lookup
        ;
   

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 

    quad* d_lookup = device_alloc<quad>(num_lookup) ; 

    QCtx_boundary_lookup_all(numBlocks, threadsPerBlock, d_ctx, d_lookup, width, height );  

    copy_device_to_host<quad>( lookup, d_lookup, num_lookup ); 

    LOG(LEVEL) << "]" ; 

}



extern "C" void QCtx_boundary_lookup_line(    dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k ); 
void QCtx::boundary_lookup_line( quad* lookup, float* domain, unsigned num_lookup, unsigned line, unsigned k ) 
{
    LOG(LEVEL) 
        << "[" 
        << " num_lookup " << num_lookup
        << " line " << line 
        << " k " << k 
        ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_lookup, 1  ); 

    float* d_domain = device_alloc<float>(num_lookup) ; 

    copy_host_to_device<float>( d_domain, domain, num_lookup ); 

    quad* d_lookup = device_alloc<quad>(num_lookup) ; 

    QCtx_boundary_lookup_line(numBlocks, threadsPerBlock, d_ctx, d_lookup, d_domain, num_lookup, line, k );  

    copy_device_to_host<quad>( lookup, d_lookup, num_lookup ); 

    device_free<float>( d_domain ); 


    LOG(LEVEL) << "]" ; 
}






unsigned QCtx::getBoundaryTexWidth() const 
{
    return bnd->tex->width ; 
}
unsigned QCtx::getBoundaryTexHeight() const 
{
    return bnd->tex->height ; 
}
const NPY<float>* QCtx::getBoundaryTexSrc() const
{
    return bnd->src ; 
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





