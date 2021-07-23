
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
#include "QProp.hh"
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

    QScint* qscint = MakeScint(slib); // custom high-definition inverse CDF for scintillation generation
    LOG(LEVEL) << qscint->desc(); 

    QBnd* qbnd = new QBnd(blib); // boundary texture with standard domain, used for standard fast property lookup 
    LOG(LEVEL) << qbnd->desc(); 

    QProp* qprop = new QProp ;  // property interpolation with per-property domains, eg used for Cerenkov RINDEX sampling 
    LOG(LEVEL) << qprop->desc(); 

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
    prop(QProp::Get()),
    ctx(new qctx),
    d_ctx(nullptr)
{
    INSTANCE = this ; 
    init(); 
}

/**
QCtx::init
------------

NB .ctx (qctx.h) is a host side instance that is populated
with device side pointers and handles and then uploaded 
to the device d_ctx.

Many device pointers and handles are then accessible from 
the qctx.h instance at the cost of only a single launch 
parameter argument: the d_ctx pointer.

The advantage of this approach is it avoids kernel 
launches having very long argument lists and provides a natural 
place (qctx.h) to add GPU side functionality. 

**/
void QCtx::init()
{
    LOG(LEVEL) 
        << " rng " << rng 
        << " scint " << scint
        << " bnd " << bnd
        << " prop " << prop
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
        ctx->boundary_tex_MaterialLine_Water = bnd->getMaterialLine("Water") ; 
        ctx->boundary_tex_MaterialLine_LS    = bnd->getMaterialLine("LS") ; 
    } 

    if(prop)
    {
        LOG(LEVEL) << " prop " << prop->desc() ; 
        ctx->prop = prop->getDevicePtr() ; 
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


void QCtx::configureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
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
}

template<typename T>
void QCtx::copy_device_to_host_and_free( T* h, T* d,  unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( h ), d , size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d) ); 
}

template<typename T>
void QCtx::copy_host_to_device( T* d, const T* h, unsigned num_items)
{
    size_t size = num_items*sizeof(T) ; 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d ), h , size, cudaMemcpyHostToDevice )); 
}



extern "C" void QCtx_rng_sequence_0(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* rs, unsigned num_items ); 
void QCtx::rng_sequence_0( float* rs, unsigned num_items )
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_items, 1 ); 

    float* d_rs = device_alloc<float>(num_items ); 

    QCtx_rng_sequence_0(numBlocks, threadsPerBlock, d_ctx, d_rs, num_items );  

    copy_device_to_host_and_free<float>( rs, d_rs, num_items ); 

    LOG(LEVEL) << "]" ; 
}

/**
QCtx::rng_sequence mass production with multiple launches...
--------------------------------------------------------------

Split output files too ?::

    epsilon:opticks blyth$ np.py *.npy 
    a :                                            TRngBufTest_0.npy :      (10000, 16, 16) : 8f9b27c9416a0121574730baa742b5c9 : 20210715-1227 
    epsilon:opticks blyth$ du -h TRngBufTest_0.npy
     20M	TRngBufTest_0.npy

    In [6]: (16*16*4*2*10000)/1e6
    Out[6]: 20.48

Upping to 1M would be 100x 20M = 2000M  2GB

* using floats would half storage to 1GB, just promote to double in cks/OpticksRandom::flat 
  THIS MAKES SENSE AS curand_uniform IS GENERATING float anyhow 
* 100k blocks would mean 10*100Mb files for 1M : can do in 10 launches to work on any GPU 

**/


extern "C" void QCtx_rng_sequence_f(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float*  seq, unsigned ni, unsigned nj, unsigned ioffset ); 
extern "C" void QCtx_rng_sequence_d(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, double* seq, unsigned ni, unsigned nj, unsigned ioffset ); 



/**
QCtx::rng_sequence_
---------------------

This is a workaround for not being able to template extern C symbols 
using template specialization for float and double.
With useful implementation only in the template specializations.

**/

template <typename T>
void QCtx::rng_sequence_( dim3 numblocks, dim3 threadsPerBlock, qctx* d_ctx, T* d_seq, unsigned ni_tranche, unsigned nv, unsigned ioffset )
{
    assert(0); 
}
template<>
void QCtx::rng_sequence_<float>(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* d_seq, unsigned ni_tranche, unsigned nv, unsigned ioffset )
{
    QCtx_rng_sequence_f( numBlocks, threadsPerBlock, d_ctx, d_seq, ni_tranche, nv, ioffset );     
}
template<>
void QCtx::rng_sequence_<double>(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, double* d_seq, unsigned ni_tranche, unsigned nv, unsigned ioffset )
{
    QCtx_rng_sequence_d( numBlocks, threadsPerBlock, d_ctx, d_seq, ni_tranche, nv, ioffset );     
}





template <typename T> char QCtx::typecode(){ return '?' ; }  // static 
template <> char QCtx::typecode<float>(){  return 'f' ; }
template <> char QCtx::typecode<double>(){ return 'd' ; }




template <typename T>
std::string QCtx::rng_sequence_name(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset ) // static 
{
    std::stringstream ss ; 
    ss << prefix
       << "_" << typecode<T>()
       << "_ni" << ni 
       << "_nj" << nj 
       << "_nk" << nk 
       << "_ioffset" << std::setw(6) << std::setfill('0') << ioffset 
       << ".npy"
       ; 

    std::string name = ss.str(); 
    return name ; 
}
template std::string QCtx::rng_sequence_name<float>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset ) ; 
template std::string QCtx::rng_sequence_name<double>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ioffset ) ; 



template <typename T>
std::string QCtx::rng_sequence_reldir(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size ) // static 
{
    std::stringstream ss ; 
    ss << prefix
       << "_" << typecode<T>()
       << "_ni" << ni 
       << "_nj" << nj 
       << "_nk" << nk 
       << "_tranche" << ni_tranche_size 
       ; 

    std::string reldir = ss.str(); 
    return reldir ; 
}
template std::string QCtx::rng_sequence_reldir<float>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size ) ; 
template std::string QCtx::rng_sequence_reldir<double>(const char* prefix, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size ) ; 







template <typename T>
void QCtx::rng_sequence( T* seq, unsigned ni_tranche, unsigned nv, unsigned ioffset )
{
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, ni_tranche, 1 ); 

    unsigned num_rng = ni_tranche*nv ;  

    T* d_seq = device_alloc<T>(num_rng); 

    rng_sequence_<T>(numBlocks, threadsPerBlock, d_ctx, d_seq, ni_tranche, nv, ioffset );  

    copy_device_to_host_and_free<T>( seq, d_seq, num_rng ); 
}


const char* QCtx::PREFIX = "rng_sequence" ; 

/**
QCtx::rng_sequence
---------------------

Structured reldir with appropriate name 

**/

template <typename T>
void QCtx::rng_sequence( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  )
{
    assert( ni % ni_tranche_size == 0 ); 
    unsigned num_tranche = ni/ni_tranche_size ; 
    unsigned nv = nj*nk ; 
    unsigned size = ni_tranche_size*nv ; 
    std::string reldir = rng_sequence_reldir<T>(PREFIX, ni, nj, nk, ni_tranche_size  ) ;  

    std::cout 
        << "QCtx::rng_sequence" 
        << " ni " << ni
        << " ni_tranche_size " << ni_tranche_size
        << " num_tranche " << num_tranche 
        << " reldir " << reldir.c_str()
        << " nj " << nj
        << " nk " << nk
        << " nv(nj*nk) " << nv 
        << " size(ni_tranche_size*nv) " << size 
        << " typecode " << typecode<T>() 
        << std::endl 
        ; 

    NPY<T>* seq = NPY<T>::make(ni_tranche_size, nj, nk) ; 
    seq->zero(); 
    T* values = seq->getValues(); 

    for(unsigned t=0 ; t < num_tranche ; t++)
    {
        unsigned ioffset = ni_tranche_size*t ; 
        std::string name = rng_sequence_name<T>(PREFIX, ni_tranche_size, nj, nk, ioffset ) ;  

        std::cout 
            << std::setw(3) << t 
            << std::setw(10) << ioffset 
            << std::setw(100) << name.c_str()
            << std::endl 
            ; 

        rng_sequence<T>( values, ni_tranche_size, nv, ioffset );  
        seq->save(dir, reldir.c_str(), name.c_str()); 
    }
}
template void QCtx::rng_sequence<float>( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  ); 
template void QCtx::rng_sequence<double>( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  ); 




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

    copy_device_to_host_and_free<float>( wavelength, d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 
}

/**
QCtx::generate_cerenkov
-------------------------

**/

extern "C" void QCtx_generate_cerenkov_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* wavelength, unsigned num_wavelength ); 
void QCtx::generate_cerenkov( float* wavelength, unsigned num_wavelength )
{
    LOG(LEVEL) << "[ num_wavelength " << num_wavelength ;
 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_wavelength, 1 ); 

    float* d_wavelength = device_alloc<float>(num_wavelength); 

    QCtx_generate_cerenkov_wavelength(numBlocks, threadsPerBlock, d_ctx, d_wavelength, num_wavelength );  

    copy_device_to_host_and_free<float>( wavelength, d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 
}




extern "C" void QCtx_generate_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, quad4* photon, unsigned num_photon, int print_id );
void QCtx::generate_cerenkov_photon( quad4* photon, unsigned num_photon, int print_id )
{
    LOG(LEVEL) << "[ num_photon " << num_photon ;
 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, num_photon, 1 ); 

    quad4* d_photon = device_alloc<quad4>(num_photon); 

    QCtx_generate_cerenkov_photon(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon, print_id );  

    copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 

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

    copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 

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

    copy_device_to_host_and_free<quad>( lookup, d_lookup, num_lookup ); 

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

    copy_device_to_host_and_free<quad>( lookup, d_lookup, num_lookup ); 

    device_free<float>( d_domain ); 


    LOG(LEVEL) << "]" ; 
}





extern "C" void QCtx_prop_lookup( dim3 numBlocks, dim3 threadsPerBlock, qctx* d_ctx, float* lookup, const float* domain, unsigned domain_width, unsigned* pids, unsigned num_pids ); 
void QCtx::prop_lookup( float* lookup, const float* domain, unsigned domain_width, const std::vector<unsigned>& pids ) 
{
    unsigned num_pids = pids.size() ; 
    unsigned num_lookup = num_pids*domain_width ; 
    LOG(LEVEL) 
        << "[" 
        << " num_pids " << num_pids
        << " domain_width " << domain_width 
        << " num_lookup " << num_lookup
        ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, domain_width, num_pids  ); 

    float* d_domain = device_alloc<float>(domain_width) ; 
    unsigned* d_pids = device_alloc<unsigned>(num_pids) ; 

    copy_host_to_device<float>( d_domain, domain, domain_width ); 
    copy_host_to_device<unsigned>( d_pids, pids.data(), num_pids ); 

    float* d_lookup = device_alloc<float>(num_lookup) ; 

    QCtx_prop_lookup(numBlocks, threadsPerBlock, d_ctx, d_lookup, d_domain, domain_width, d_pids, num_pids );  

    copy_device_to_host_and_free<float>( lookup, d_lookup, num_lookup ); 

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


 


