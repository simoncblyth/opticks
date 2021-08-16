
#include "PLOG.hh"
#include "SSys.hh"
#include "scuda.h"

#include "NPY.hpp"
#include "GGeo.hh"
#include "GBndLib.hh"
#include "GScintillatorLib.hh"

#include "QUDA_CHECK.h"
#include "QU.hh"

#include "qrng.h"
#include "qctx.h"

#include "QRng.hh"
#include "QTex.hh"
#include "QScint.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QCtx.hh"

template <typename T>
const plog::Severity QCtx<T>::LEVEL = PLOG::EnvLevel("QCtx", "INFO"); 

template <typename T>
const QCtx<T>* QCtx<T>::INSTANCE = nullptr ; 

template <typename T>
const QCtx<T>* QCtx<T>::Get(){ return INSTANCE ; }

template <typename T>
void QCtx<T>::Init(const GGeo* ggeo)
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

    QProp<T>* qprop = new QProp<T> ;  // property interpolation with per-property domains, eg used for Cerenkov RINDEX sampling 
    LOG(LEVEL) << qprop->desc(); 


}

template <typename T>
QScint* QCtx<T>::MakeScint(const GScintillatorLib* slib)
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


template <typename T>
QCtx<T>::QCtx()
    :
    rng(QRng::Get()),
    scint(QScint::Get()),
    bnd(QBnd::Get()),
    prop(QProp<T>::Get()),
    ctx(new qctx<T>),
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
template <typename T>
void QCtx<T>::init()
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
        ctx->r = rng->qr->rng_states ; 
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

    d_ctx = QU::UploadArray<qctx<T>>(ctx, 1 );  

    LOG(LEVEL) << desc() ; 
}

template <typename T>
char QCtx<T>::getScintTexFilterMode() const 
{
    return scint->tex->getFilterMode() ; 
}


template<typename T>
std::string QCtx<T>::desc() const
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



template<typename T>
void QCtx<T>::configureLaunch(unsigned width, unsigned height ) 
{
    QU::ConfigureLaunch(numBlocks, threadsPerBlock, width, height); 
}

template<typename T>
void QCtx<T>::configureLaunch2D(unsigned width, unsigned height ) 
{
    QU::ConfigureLaunch2D(numBlocks, threadsPerBlock, width, height); 
}


template <typename T>
extern void QCtx_rng_sequence_0(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, T* rs, unsigned num_items ); 


template<typename T>
void QCtx<T>::rng_sequence_0( T* rs, unsigned num_items )
{
    configureLaunch(num_items, 1 ); 

    T* d_rs = QU::device_alloc<T>(num_items ); 

    QCtx_rng_sequence_0<T>(numBlocks, threadsPerBlock, d_ctx, d_rs, num_items );  

    QU::copy_device_to_host_and_free<T>( rs, d_rs, num_items ); 

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


template <typename T>
extern void QCtx_rng_sequence(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, T*  seq, unsigned ni, unsigned nj, unsigned ioffset ); 


template <typename T>
void QCtx<T>::rng_sequence( T* seq, unsigned ni_tranche, unsigned nv, unsigned ioffset )
{
    configureLaunch(ni_tranche, 1 ); 

    unsigned num_rng = ni_tranche*nv ;  

    T* d_seq = QU::device_alloc<T>(num_rng); 

    QCtx_rng_sequence<T>(numBlocks, threadsPerBlock, d_ctx, d_seq, ni_tranche, nv, ioffset );  

    QU::copy_device_to_host_and_free<T>( seq, d_seq, num_rng ); 
}


template <typename T>
const char* QCtx<T>::PREFIX = "rng_sequence" ; 

/**
QCtx::rng_sequence
---------------------

The ni is split into tranches of ni_tranche_size each.

**/

template <typename T>
void QCtx<T>::rng_sequence( const char* dir, unsigned ni, unsigned nj, unsigned nk, unsigned ni_tranche_size  )
{
    assert( ni % ni_tranche_size == 0 ); 
    unsigned num_tranche = ni/ni_tranche_size ; 
    unsigned nv = nj*nk ; 
    unsigned size = ni_tranche_size*nv ; 
    std::string reldir = QU::rng_sequence_reldir<T>(PREFIX, ni, nj, nk, ni_tranche_size  ) ;  

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
        << " typecode " << QU::typecode<T>() 
        << std::endl 
        ; 

    NPY<T>* seq = NPY<T>::make(ni_tranche_size, nj, nk) ; 
    seq->zero(); 
    T* values = seq->getValues(); 

    for(unsigned t=0 ; t < num_tranche ; t++)
    {
        unsigned ioffset = ni_tranche_size*t ; 
        std::string name = QU::rng_sequence_name<T>(PREFIX, ni_tranche_size, nj, nk, ioffset ) ;  

        std::cout 
            << std::setw(3) << t 
            << std::setw(10) << ioffset 
            << std::setw(100) << name.c_str()
            << std::endl 
            ; 

        rng_sequence( values, ni_tranche_size, nv, ioffset );  
        seq->save(dir, reldir.c_str(), name.c_str()); 
    }
}




/**
QCtx::scint_wavelength
----------------------------------

Setting envvar QCTX_DISABLE_HD disables multiresolution handling
and causes the returned hd_factor to be zero rather then 
the typical values of 10 or 20 which depend on the buffer creation.

**/

template <typename T>
extern void QCtx_scint_wavelength(   dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, T* wavelength, unsigned num_wavelength, unsigned hd_factor ); 

template <typename T>
void QCtx<T>::scint_wavelength( T* wavelength, unsigned num_wavelength, unsigned& hd_factor )
{
    bool qctx_disable_hd = SSys::getenvbool("QCTX_DISABLE_HD"); 
    hd_factor = qctx_disable_hd ? 0u : scint->tex->getHDFactor() ; 
    // HMM: perhaps get this from ctx rather than occupying an argument slot  
    LOG(LEVEL) << "[" << " qctx_disable_hd " << qctx_disable_hd << " hd_factor " << hd_factor ; 

    configureLaunch(num_wavelength, 1 ); 

    T* d_wavelength = QU::device_alloc<T>(num_wavelength); 

    QCtx_scint_wavelength<T>(numBlocks, threadsPerBlock, d_ctx, d_wavelength, num_wavelength, hd_factor );  

    QU::copy_device_to_host_and_free<T>( wavelength, d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 
}

/**
QCtx::cerenkov_wavelength
---------------------------

**/

template <typename T>
extern void QCtx_cerenkov_wavelength(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, T* wavelength, unsigned num_wavelength ); 

template <typename T>
void QCtx<T>::cerenkov_wavelength( T* wavelength, unsigned num_wavelength )
{
    LOG(LEVEL) << "[ num_wavelength " << num_wavelength ;
 
    configureLaunch(num_wavelength, 1 ); 

    T* d_wavelength = QU::device_alloc<T>(num_wavelength); 

    QCtx_cerenkov_wavelength(numBlocks, threadsPerBlock, d_ctx, d_wavelength, num_wavelength );  

    QU::copy_device_to_host_and_free<T>( wavelength, d_wavelength, num_wavelength ); 

    LOG(LEVEL) << "]" ; 
}




template <typename T>
extern void QCtx_cerenkov_photon(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, quad4* photon, unsigned num_photon, int print_id );


template <typename T>
void QCtx<T>::cerenkov_photon( quad4* photon, unsigned num_photon, int print_id )
{
    LOG(LEVEL) << "[ num_photon " << num_photon ;
 
    configureLaunch(num_photon, 1 ); 

    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 

    QCtx_cerenkov_photon<T>(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon, print_id );  

    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 

    LOG(LEVEL) << "]" ; 
}



template <typename T>
extern void QCtx_cerenkov_photon_enprop(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, quad4* photon, unsigned num_photon, int print_id );


template <typename T>
void QCtx<T>::cerenkov_photon_enprop( quad4* photon, unsigned num_photon, int print_id )
{
    LOG(LEVEL) << "[ num_photon " << num_photon ;
 
    configureLaunch(num_photon, 1 ); 

    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 

    QCtx_cerenkov_photon_enprop<T>(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon, print_id );  

    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 

    LOG(LEVEL) << "]" ; 
}




template <typename T>
extern void QCtx_cerenkov_photon_expt(dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, quad4* photon, unsigned num_photon, int print_id );


template <typename T>
void QCtx<T>::cerenkov_photon_expt( quad4* photon, unsigned num_photon, int print_id )
{
    LOG(LEVEL) << "[ num_photon " << num_photon ;
 
    configureLaunch(num_photon, 1 ); 

    quad4* d_photon = QU::device_alloc<quad4>(num_photon); 

    QCtx_cerenkov_photon_expt<T>(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon, print_id );  

    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 

    LOG(LEVEL) << "]" ; 
}





template <typename T>
void QCtx<T>::dump_wavelength( T* wavelength, unsigned num_wavelength, unsigned edgeitems )
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


template <typename T>
extern void QCtx_scint_photon( dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, quad4* photon , unsigned num_photon ); 


template <typename T>
void QCtx<T>::scint_photon( quad4* photon, unsigned num_photon )
{
    LOG(LEVEL) << "[" ; 

    configureLaunch( num_photon, 1 ); 

    quad4* d_photon = QU::device_alloc<quad4>(num_photon) ; 

    QCtx_scint_photon(numBlocks, threadsPerBlock, d_ctx, d_photon, num_photon );  

    QU::copy_device_to_host_and_free<quad4>( photon, d_photon, num_photon ); 

    LOG(LEVEL) << "]" ; 
}


template <typename T>
extern void QCtx_boundary_lookup_all(    dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, quad* lookup  , unsigned width, unsigned height ); 

template <typename T>
void QCtx<T>::boundary_lookup_all(quad* lookup, unsigned width, unsigned height )
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
   

    configureLaunch(width, height ); 

    quad* d_lookup = QU::device_alloc<quad>(num_lookup) ; 

    QCtx_boundary_lookup_all(numBlocks, threadsPerBlock, d_ctx, d_lookup, width, height );  

    QU::copy_device_to_host_and_free<quad>( lookup, d_lookup, num_lookup ); 

    LOG(LEVEL) << "]" ; 

}

template <typename T>
extern void QCtx_boundary_lookup_line(    dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k ); 


template <typename T>
void QCtx<T>::boundary_lookup_line( quad* lookup, T* domain, unsigned num_lookup, unsigned line, unsigned k ) 
{
    LOG(LEVEL) 
        << "[" 
        << " num_lookup " << num_lookup
        << " line " << line 
        << " k " << k 
        ; 

    configureLaunch(num_lookup, 1  ); 

    T* d_domain = QU::device_alloc<T>(num_lookup) ; 

    QU::copy_host_to_device<T>( d_domain, domain, num_lookup ); 

    quad* d_lookup = QU::device_alloc<quad>(num_lookup) ; 

    QCtx_boundary_lookup_line<T>(numBlocks, threadsPerBlock, d_ctx, d_lookup, d_domain, num_lookup, line, k );  

    QU::copy_device_to_host_and_free<quad>( lookup, d_lookup, num_lookup ); 

    QU::device_free<T>( d_domain ); 


    LOG(LEVEL) << "]" ; 
}





/**
QCtx::prop_lookup
--------------------

suspect problem when have fine domain and many pids due to 2d launch config,
BUT when have 1d launch there is no problem to launch millions of threads : hence the 
below *prop_lookup_onebyone* 

**/


template <typename T>
extern void QCtx_prop_lookup( dim3 numBlocks, dim3 threadsPerBlock, qctx<T>* d_ctx, T* lookup, const T* domain, unsigned domain_width, unsigned* pids, unsigned num_pids ); 

template <typename T>
void QCtx<T>::prop_lookup( T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids ) 
{
    unsigned num_pids = pids.size() ; 
    unsigned num_lookup = num_pids*domain_width ; 
    LOG(LEVEL) 
        << "[" 
        << " num_pids " << num_pids
        << " domain_width " << domain_width 
        << " num_lookup " << num_lookup
        ; 

    configureLaunch(domain_width, num_pids  ); 

    unsigned* d_pids = QU::device_alloc<unsigned>(num_pids) ; 
    T* d_domain = QU::device_alloc<T>(domain_width) ; 
    T* d_lookup = QU::device_alloc<T>(num_lookup) ; 

    QU::copy_host_to_device<T>( d_domain, domain, domain_width ); 
    QU::copy_host_to_device<unsigned>( d_pids, pids.data(), num_pids ); 

    QCtx_prop_lookup(numBlocks, threadsPerBlock, d_ctx, d_lookup, d_domain, domain_width, d_pids, num_pids );  

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup ); 
    QU::device_free<T>( d_domain ); 
    QU::device_free<unsigned>( d_pids ); 

    LOG(LEVEL) << "]" ; 
}



/**
Hmm doing lookups like this is a very common pattern, could do with 
a sub context to carry the pieces to simplify doing that.
**/

template <typename T>
extern void QCtx_prop_lookup_one(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qctx<T>* ctx, 
    T* lookup, 
    const T* domain, 
    unsigned domain_width, 
    unsigned num_pids, 
    unsigned pid, 
    unsigned ipid 
);

template <typename T>
void QCtx<T>::prop_lookup_onebyone( T* lookup, const T* domain, unsigned domain_width, const std::vector<unsigned>& pids ) 
{
    unsigned num_pids = pids.size() ; 
    unsigned num_lookup = num_pids*domain_width ; 
    LOG(LEVEL) 
        << "[" 
        << " num_pids " << num_pids
        << " domain_width " << domain_width 
        << " num_lookup " << num_lookup
        ; 

    configureLaunch(domain_width, 1  ); 

    T* d_domain = QU::device_alloc<T>(domain_width) ; 
    QU::copy_host_to_device<T>( d_domain, domain, domain_width ); 

    T* d_lookup = QU::device_alloc<T>(num_lookup) ; 

    // separate launches for each pid
    for(unsigned ipid=0 ; ipid < num_pids ; ipid++)
    {
        unsigned pid = pids[ipid] ; 
        QCtx_prop_lookup_one<T>(numBlocks, threadsPerBlock, d_ctx, d_lookup, d_domain, domain_width, num_pids, pid, ipid );  
    }

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup ); 

    QU::device_free<T>( d_domain ); 

    LOG(LEVEL) << "]" ; 
}




template <typename T>
unsigned QCtx<T>::getBoundaryTexWidth() const 
{
    return bnd->tex->width ; 
}
template <typename T>
unsigned QCtx<T>::getBoundaryTexHeight() const 
{
    return bnd->tex->height ; 
}
template <typename T>
const NPY<float>* QCtx<T>::getBoundaryTexSrc() const
{
    return bnd->src ; 
}

template <typename T>
void QCtx<T>::dump_photon( quad4* photon, unsigned num_photon, unsigned edgeitems )
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


template struct QCtx<float> ; 
template struct QCtx<double> ;

 

