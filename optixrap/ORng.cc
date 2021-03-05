/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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


const plog::Severity ORng::LEVEL = PLOG::EnvLevel("ORng", "DEBUG") ; 

ORng::ORng(Opticks* ok, OContext* ocontext) 
    :
    m_ok(ok),
    m_mask(ok->getMask()),
    m_ocontext(ocontext),
    m_context(m_ocontext->getContext()),
    m_rng_wrapper(NULL),
    m_rng_skipahead(0)   
{
   init();
}

/**
ORng::init
-------------

Formerly used INPUT_OUTPUT m_rng_states but Opticks does the full simulation
in one kernel call so there is no need to persist curandState into global buffer, 
the curandState is copied into registers and updated by curand_uniform calls::

   m_rng_states = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);

The use of LoadIntoHostBuffer looks a bit perplexing but contrast with OContext::upload
using memcpy to copy into a mapped host pointer is the standard way to load into an 
OptiX buffer. However this can be done with one less GPU buffer by using interop to 
get OptiX to adopt the CUDA buffer which already exists. 


masked running
~~~~~~~~~~~~~~~~~

For masked running the LoadIntoHostBufferMasked fabricates an OptiX buffer
with just the curandStates needes for the mask list of photon indices.


**/

void ORng::init()
{
    unsigned rng_max = m_ok->getRngMax();
    if(rng_max == 0 )
    {
        LOG(error) 
            << " EARLY EXIT "
            << " rng_max " << rng_max
            ;
        return ;
    }
    const char* RNGDir = m_ok->getRNGDir();
    unsigned num_mask = m_mask.size() ; 

    LOG(LEVEL) 
        << " rng_max " << rng_max
        << " RNGDir " << RNGDir
        << " num_mask " << num_mask
        ;

    unsigned long long seed = 0ull ; 
    unsigned long long offset = 0ull ; 

    // these are the defaults that have been in use for years... 
    // they should be adjusted depending on the GPU 
    // BUT this will have no impact here, it is only relevant when  
    // creating the curandState buffer with cudarap-prepare-installcache
    // HMM : this API needs overhaul as it gives the wrong impression of 
    // what is possible  

    unsigned max_blocks = 128 ; 
    unsigned threads_per_block = 256 ; 

    bool verbose = false ; 

    m_rng_wrapper = cuRANDWrapper::instanciate( rng_max, RNGDir, seed, offset, max_blocks, threads_per_block, verbose );

    // OptiX owned RNG states buffer (not CUDA owned)
    m_rng_states = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER);      

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
    m_context["rng_skipahead"]->setUint(m_rng_skipahead) ; 
}

void ORng::setSkipAhead( unsigned skipahead )
{
    LOG(LEVEL) << " skipahead " << skipahead ; 
    m_rng_skipahead = skipahead ; 
    m_context["rng_skipahead"]->setUint(m_rng_skipahead) ; 
}
unsigned ORng::getSkipAhead() const 
{
    return m_rng_skipahead ; 
}

