#include <cuda_runtime.h>
#include <sstream>

#include "SStr.hh"
#include "scuda.h"
#include "QUDA_CHECK.h"
#include "NP.hh"
#include "QCtx.hh"
#include "QProp.hh"
#include "QU.hh"
#include "qprop.h"
#include "PLOG.hh"

const plog::Severity QProp::LEVEL = PLOG::EnvLevel("QProp", "INFO"); 

const QProp* QProp::INSTANCE = nullptr ; 
const QProp* QProp::Get(){ return INSTANCE ; }

QProp::QProp(const NP* a_)
    :
    a(a_),
    pp(a->cvalues<float>()),
    nv(a->num_values()),
    ni(a->shape[0]),
    nj(a->shape[1]),
    prop(new qprop),
    d_prop(nullptr)
{
    INSTANCE = this ; 
    init(); 
} 

void QProp::init()
{
    assert( a->uifc == 'f' ); 
    assert( a->ebyte == 4 );  
    assert( a->shape.size() == 2 ); 

    dump(); 
    uploadProps(); 
}

void QProp::uploadProps()
{
    prop->pp = QCtx::device_alloc<float>(nv) ; 
    prop->height = ni ; 
    prop->width =  nj ; 

    QCtx::copy_host_to_device<float>( prop->pp, pp, nv ); 

    d_prop = QU::UploadArray<qprop>(prop, 1 );  
}

void QProp::dump()
{
    UIF u1, u2 ;
  
    for(unsigned i=0 ; i < ni ; i++)
    {
        for(unsigned j=0 ; j < nj ; j++)
        {
            std::cout 
                << std::setw(10) << std::fixed << std::setprecision(5) << pp[nj*i+j] << " " 
                ; 
        }
    
        u1.f = pp[nj*i+nj-1] ; 
        u2.f = pp[nj*i+nj-2] ; 

        unsigned prop_ni  = u1.u ; 
        unsigned prop_idx = u2.u ; 

        std::cout 
            << " prop_idx:" << std::setw(5) << prop_idx
            << " prop_ni :" << std::setw(5) << prop_ni 
            << " nj/2  :" << std::setw(5) << nj/2 
            << std::endl
            ; 

        assert( prop_ni < nj/2 ) ;  // factor 2 as domain and values are interleaved for each property
    }
}



extern "C" void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qprop* prop, 
    float* lookup, 
    const float* domain, 
    unsigned lookup_prop, 
    unsigned domain_width
); 

void QProp::lookup( float* lookup, const float* domain,  unsigned lookup_prop, unsigned domain_width )
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
    configureLaunch( numBlocks, threadsPerBlock, lookup_prop, domain_width ); 

    QProp_lookup(numBlocks, threadsPerBlock, d_prop, d_lookup, d_domain, lookup_prop, domain_width );  

    QCtx::copy_device_to_host_and_free<float>( lookup, d_lookup, num_lookup ); 
     
    LOG(LEVEL) << "]" ; 
}


void QProp::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

