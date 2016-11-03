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


ORng::ORng(Opticks* ok, OContext* ocontext) 
   :
   m_ok(ok),
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

    LOG(debug) << "ORng::init"
               << " rng_max " << rng_max
               << " rngCacheDir " << rngCacheDir
               ;

    m_rng_wrapper = cuRANDWrapper::instanciate( rng_max, rngCacheDir );

    // OptiX owned RNG states buffer (not CUDA owned)
    m_rng_states = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_rng_states->setElementSize(sizeof(curandState));
    m_rng_states->setSize(rng_max);
    m_context["rng_states"]->setBuffer(m_rng_states);


    {
        curandState* host_rng_states = static_cast<curandState*>( m_rng_states->map() );

        m_rng_wrapper->setItems(rng_max);
        m_rng_wrapper->fillHostBuffer(host_rng_states, rng_max);

        m_rng_states->unmap();
    }

    //
    // TODO: investigate Thrust based alternatives for curand initialization 
    //       potential for eliminating cudawrap- 
    //
}




