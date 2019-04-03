#include "PLOG.hh"
#include "Opticks.hh"

#include "OContext.hh"
#include "ORng.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>
using namespace optix ; 

// cudawrap-  NB needs to be after namespace optix
#include "cuRANDWrapper.hh"


const plog::Severity ORng::LEVEL = debug ; 

ORng::ORng(Opticks* ok, OContext* ocontext) 
   :
   m_ok(ok),
   m_mask(ok->getMask()),
   m_ocontext(ocontext),
   m_context(m_ocontext->getContext()),
   m_rng_wrapper(NULL)
{
   init();
}

void ORng::init()
{
    unsigned int rng_max = m_ok->getRngMax();
    if(rng_max == 0 )
    {
        LOG(warning) << "ORng::init"   
                     << " EARLY EXIT "
                     << " rng_max " << rng_max
                     ;
        return ;
    }
    const char* rngCacheDir = m_ok->getRNGInstallCacheDir();
    unsigned num_mask = m_mask.size() ; 

    LOG(LEVEL) << "ORng::init"
               << " rng_max " << rng_max
               << " rngCacheDir " << rngCacheDir
               << " num_mask " << num_mask
               ;

    m_rng_wrapper = cuRANDWrapper::instanciate( rng_max, rngCacheDir );

    // OptiX owned RNG states buffer (not CUDA owned)
    m_rng_states = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);

    m_rng_states->setElementSize(sizeof(curandState));

    if(num_mask == 0)
    {
        m_rng_states->setSize(rng_max);

        curandState* host_rng_states = static_cast<curandState*>( m_rng_states->map() );

        m_rng_wrapper->setItems(rng_max); // why ? to identify which cache file to load i suppose

        m_rng_wrapper->LoadIntoHostBuffer(host_rng_states, rng_max );

        m_rng_states->unmap();
    }
    else
    {
        m_rng_states->setSize(num_mask);

        curandState* host_rng_states = static_cast<curandState*>( m_rng_states->map() );

        m_rng_wrapper->setItems(rng_max); // still need to load the full cache

        m_rng_wrapper->LoadIntoHostBufferMasked(host_rng_states, m_mask ) ; // but make partial copy 

        m_rng_states->unmap();
    }

    m_context["rng_states"]->setBuffer(m_rng_states);
}


