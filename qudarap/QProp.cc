#include <cuda_runtime.h>
#include <sstream>

#include "SStr.hh"
#include "scuda.h"
#include "QUDA_CHECK.h"
#include "NP.hh"
#include "QCtx.hh"
#include "QProp.hh"
#include "PLOG.hh"

const plog::Severity QProp::LEVEL = PLOG::EnvLevel("QProp", "INFO"); 

const QProp* QProp::INSTANCE = nullptr ; 
const QProp* QProp::Get(){ return INSTANCE ; }

QProp::QProp(const NP* prop_)
    :
    prop(prop_),
    pp(prop->cvalues<float>()),
    nv(prop->num_values()),
    ni(prop->shape[0]),
    nj(prop->shape[1]),
    d_pp(nullptr)
{
    INSTANCE = this ; 
    init(); 
} 

void QProp::init()
{
    assert( prop->uifc == 'f' ); 
    assert( prop->ebyte == 4 );  
    assert( prop->shape.size() == 2 ); 

    dump(); 
    upload(); 
}

void QProp::upload()
{
    d_pp = QCtx::device_alloc<float>(nv) ; 
    QCtx::copy_host_to_device<float>( d_pp, pp, nv ); 
}

void QProp::clear()
{
    QCtx::device_free<float>( d_pp );  
    d_pp = nullptr ; 
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

        unsigned pp_ni  = u1.u ; 
        unsigned pp_idx = u2.u ; 

        std::cout 
            << " pp_idx:" << std::setw(5) << pp_idx
            << " pp_ni :" << std::setw(5) << pp_ni 
            << " nj/2  :" << std::setw(5) << nj/2 
            << std::endl
            ; 

        assert( pp_ni < nj/2 ) ;
    }
}


void QProp::lookup(float x0, float x1, unsigned nx)
{
   // hmm this makes more sense to be in QPropTest  
    NP* x = NP::Linspace<float>( x0, x1, nx ); 
    NP* y = NP::Make<float>(ni, nx ); 

    lookup(y->values<float>(), x->cvalues<float>(), ni, nx ); 
}

extern "C" void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    float* lookup, 
    const float* domain, 
    unsigned lookup_prop, 
    unsigned domain_width, 
    const float* pp, 
    unsigned pp_height, 
    unsigned pp_width 
); 

void QProp::lookup( float* lookup, const float* domain,  unsigned lookup_prop, unsigned domain_width )
{
    unsigned num_lookup = lookup_prop*domain_width ; 

    LOG(LEVEL) 
        << "["
        << " lookup_prop " << lookup_prop
        << " domain_width " << domain_width
        << " num_lookup " << num_lookup
        << " ni " << ni
        << " nj " << nj
        ; 

    float* d_domain = QCtx::device_alloc<float>(domain_width) ; 
    QCtx::copy_host_to_device<float>( d_domain, domain, domain_width  ); 

    float* d_lookup = QCtx::device_alloc<float>(num_lookup) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, lookup_prop, domain_width ); 

    QProp_lookup(numBlocks, threadsPerBlock, d_lookup, d_domain, lookup_prop, domain_width, d_pp, ni, nj );  

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

