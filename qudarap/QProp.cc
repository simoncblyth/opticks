#include <cuda_runtime.h>
#include <sstream>

#include "SStr.hh"
#include "SPath.hh"
#include "scuda.h"
#include "QUDA_CHECK.h"
#include "NP.hh"
#include "QCtx.hh"
#include "QProp.hh"
#include "QU.hh"
#include "qprop.h"
#include "PLOG.hh"

const plog::Severity QProp::LEVEL = PLOG::EnvLevel("QProp", "INFO"); 

//const char* QProp::DEFAULT_PATH = "/tmp/np/test_compound_np_interp.npy" ; 
const char* QProp::DEFAULT_PATH = "$OPTICKS_KEYDIR/GScintillatorLib/LS_ori/RINDEX.npy" ;

const QProp* QProp::INSTANCE = nullptr ; 
const QProp* QProp::Get(){ return INSTANCE ; }

qprop* QProp::getDevicePtr() const
{
    return d_prop ; 
}


/**
QProp::Load
--------------

Mockup a real set of multiple properties

**/

const NP* QProp::Load(const char* path_ )  // static 
{
    const char* path = SPath::Resolve(path_); 
    LOG(LEVEL) 
        << "path_ " << path_  
        << "path " << path  
        ;

    if( path == nullptr ) return nullptr ; 
    NP* a = NP::Load(path) ; 
    assert( strcmp( a->dtype, "<f8") == 0 ); 
    a->pscale<double>(1e6, 0u);   // energy scale from MeV to eV,   1.55 to 15.5 eV

    NP* b = NP::Load(path); 
    b->pscale<double>(1e6, 0u); 
    b->pscale<double>(1.05, 1u); 

    NP* c = NP::Load(path); 
    c->pscale<double>(1e6, 0u); 
    c->pscale<double>(0.95, 1u); 

    NP* an = NP::MakeNarrow( a );
    NP* bn = NP::MakeNarrow( b );
    NP* cn = NP::MakeNarrow( c );

    std::vector<const NP*> aa = {an, bn, cn} ;
    NP* com = NP::Combine(aa) ; 
    LOG(LEVEL) 
        << " com " << ( com ? com->desc() : "-" )
        ;

    return com ; 
}

QProp::QProp(const char* path_)
    :
    path(path_ ? strdup(path_) : DEFAULT_PATH),
    a(Load(path)),
    pp(a ? a->cvalues<float>() : nullptr),
    nv(a ? a->num_values() : 0),
    ni(a ? a->shape[0] : 0 ),
    nj(a ? a->shape[1] : 0 ),
    nk(a ? a->shape[2] : 0 ),
    prop(new qprop),
    d_prop(nullptr)
{
    INSTANCE = this ; 
    init(); 
} 

QProp::~QProp()
{
    QUDA_CHECK(cudaFree(prop->pp)); 
    QUDA_CHECK(cudaFree(d_prop)); 
}

std::string QProp::desc() const 
{
    std::stringstream ss ; 
    ss << "QProp::desc"
       << " path " << ( path ? path : "-" ) 
       << " a " << ( a ? a->desc() : "-" )
       << " nv " << nv
       << " ni " << ni
       << " nj " << nj
       << " nk " << nk
       ;
    return ss.str(); 
}

void QProp::init()
{
    assert( a->uifc == 'f' ); 
    assert( a->ebyte == 4 );  
    assert( a->shape.size() == 3 ); 

    //dump(); 
    uploadProps(); 
}

void QProp::uploadProps()
{
    prop->pp = QCtx::device_alloc<float>(nv) ; 
    prop->height = ni ; 
    prop->width =  nj*nk ; 

    QCtx::copy_host_to_device<float>( prop->pp, pp, nv ); 

    d_prop = QU::UploadArray<qprop>(prop, 1 );  
}

void QProp::dump() const 
{
    LOG(info) << desc() ; 
    UIF u  ;
    for(unsigned i=0 ; i < ni ; i++)
    {
        for(unsigned j=0 ; j < nj ; j++)
        {
            for(unsigned k=0 ; k < nk ; k++)
            {
                std::cout 
                    << std::setw(10) << std::fixed << std::setprecision(5) << pp[nk*nj*i+j*nk+k] << " " 
                    ; 
            }
    
            u.f = pp[nk*nj*i+j*nk+nk-1] ; 
            unsigned prop_ni  = u.u ; 
            std::cout 
                << " prop_ni :" << std::setw(5) << prop_ni 
                << std::endl
                ; 

            assert( prop_ni < nj ) ;
        }
    }
}



extern "C" void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qprop* prop, 
    float* lookup, 
    const float* domain, 
    unsigned iprop, 
    unsigned domain_width
); 

void QProp::lookup( float* lookup, const float* domain,  unsigned lookup_prop, unsigned domain_width ) const 
{
    unsigned num_lookup = lookup_prop*domain_width ; 

    LOG(LEVEL) 
        << "["
        << " lookup_prop " << lookup_prop
        << " domain_width " << domain_width
        << " num_lookup " << num_lookup
        ; 

    float* d_domain = QCtx::device_alloc<float>(domain_width) ; 
    QCtx::copy_host_to_device<float>( d_domain, domain, domain_width  ); 

    float* d_lookup = QCtx::device_alloc<float>(num_lookup) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, domain_width, 1 ); 

    for(unsigned iprop=0 ; iprop < lookup_prop ; iprop++)
    {
        QProp_lookup(numBlocks, threadsPerBlock, d_prop, d_lookup, d_domain, iprop, domain_width );  
    }

    QCtx::copy_device_to_host_and_free<float>( lookup, d_lookup, num_lookup ); 
     
    LOG(LEVEL) << "]" ; 
}


void QProp::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ) const 
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

