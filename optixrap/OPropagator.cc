#include "OPropagator.hh"

// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"

// opticksgeo-
#include "OpticksHub.hh"


// optixrap-
#include "OContext.hh"
#include "OConfig.hh"
#include "STimes.hh"
#include "OEvent.hh"

#include "OBuf.hh"
#include "OBufPair.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>
using namespace optix ; 

// brap-
#include "timeutil.hh"

// npy-
#include "GLMPrint.hpp"
#include "NPY.hpp"


// cudawrap-  NB needs to be after namespace optix
#include "cuRANDWrapper.hh"


#include "PLOG.hh"




void OPropagator::setOverride(unsigned int override_)
{
    m_override = override_ ; 
}
void OPropagator::setEntry(unsigned int entry_index)
{
    m_entry_index = entry_index;
}



OBuf* OPropagator::getSequenceBuf()
{
    return m_oevt->getSequenceBuf() ;
}
OBuf* OPropagator::getPhotonBuf()
{
    return m_oevt->getPhotonBuf() ;
}
OBuf* OPropagator::getGenstepBuf()
{
    return m_oevt->getGenstepBuf() ;
}
OBuf* OPropagator::getRecordBuf()
{
    return m_oevt->getRecordBuf() ; 
}



STimes* OPropagator::getPrelaunchTimes()
{
    return m_prelaunch_times ; 
}
STimes* OPropagator::getLaunchTimes()
{
    return m_launch_times ; 
}



OPropagator::OPropagator(OContext* ocontext, OpticksHub* hub, unsigned entry, int override_) 
   :
    m_ocontext(ocontext),
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_zero(hub->getZeroEvent()),
    m_oevt(new OEvent(m_ocontext, m_zero)),
    m_prelaunch_times(new STimes),
    m_launch_times(new STimes),
    m_prelaunch(false),
    m_entry_index(entry),

    m_rng_wrapper(NULL),
    m_count(0),
    m_width(0),
    m_height(0),
    m_prep(0),
    m_time(0),
    m_override(override_)
{
    init();
}


void OPropagator::init()
{
    initParameters();
    initRng();

    prelaunch();
}


void OPropagator::initParameters()
{
    m_context = m_ocontext->getContext();

    m_context[ "propagate_epsilon"]->setFloat( m_ok->getEpsilon() );  // TODO: check impact of changing propagate_epsilon
    m_context[ "bounce_max" ]->setUint( m_ok->getBounceMax() );
    m_context[ "record_max" ]->setUint( m_ok->getRecordMax() );

    optix::uint4 debugControl = optix::make_uint4(m_ocontext->getDebugPhoton(),0,0,0);
    LOG(debug) << "OPropagator::init debugControl " 
              << " x " << debugControl.x 
              << " y " << debugControl.y
              << " z " << debugControl.z 
              << " w " << debugControl.w 
              ;

    m_context["debug_control"]->setUint(debugControl); 
 
    const glm::vec4& ce = m_ok->getSpaceDomain();
    const glm::vec4& td = m_ok->getTimeDomain();

    m_context["center_extent"]->setFloat( make_float4( ce.x, ce.y, ce.z, ce.w ));
    m_context["time_domain"]->setFloat(   make_float4( td.x, td.y, td.z, td.w ));
}


void OPropagator::initRng()
{
    unsigned int rng_max = m_ok->getRngMax();
    if(rng_max == 0 )
    {
        LOG(warning) << "OPropagator::initRng"   
                     << " EARLY EXIT "
                     << " rng_max " << rng_max
                     ;
        return ;
    }
    const char* rngCacheDir = m_ok->getRNGInstallCacheDir();

    LOG(info) << "OPropagator::initRng"
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




void OPropagator::initEvent()
{
    OpticksEvent* evt = m_hub->getOKEvent();

    if(!evt) return ;

    unsigned int numPhotons = evt->getNumPhotons();

    bool enoughRng = numPhotons <= m_ok->getRngMax() ;

    if(!enoughRng)
    {
        LOG(info) << "OPropagator::initEvent"
                  << " not enoughRng "
                  << " numPhotons " << numPhotons 
                  << " rngMax " << m_ok->getRngMax()
                  ;  
    } 

    assert( enoughRng  && "Use ggeoview-rng-prep to prepare RNG states up to the maximal number of photons to be generated per invokation " );


    m_width  = numPhotons ;
    m_height = 1 ;

    LOG(info) << "OPropagator::initEvent count " << m_count << " size(" <<  m_width << "," <<  m_height << ")";

    if(m_override > 0)
    {
        m_width = m_override ; 
        LOG(warning) << "OPropagator::initEvent OVERRIDE photon count for debugging to " << m_width ; 
    }


    if( m_ocontext->isCompute() )
    {
        m_oevt->upload(evt);
    }
    else
    {
        m_oevt = new OEvent(m_ocontext, evt) ;  // leaking resources in interop
    }
}
















void OPropagator::prelaunch()
{
    bool entry = m_entry_index > -1 ; 
    if(!entry) LOG(fatal) << "OPropagator::prelaunch MISSING entry " ;
    assert(entry);

    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  0, 0, m_prelaunch_times); 
    m_count += 1 ; 
    m_prelaunch = true ; 
}

void OPropagator::launch()
{
    assert(m_prelaunch && "must prelaunch before launch");
    m_ocontext->launch( OContext::LAUNCH,  m_entry_index,  m_width, m_height, m_launch_times);
}


void OPropagator::dumpTimes(const char* msg)
{
    LOG(info) << msg ; 
    LOG(info) << m_prelaunch_times->description("prelaunch_times");
    LOG(info) << m_launch_times->description("launch_times");
}


void OPropagator::downloadEvent()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ;
    m_oevt->download(evt);
}

void OPropagator::downloadPhotonData()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ;
    m_oevt->download(evt, OEvent::PHOTON);
}


