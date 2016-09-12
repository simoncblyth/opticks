
#include "SLog.hh"
#include "STimes.hh"

// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"
#include "OpticksCfg.hh"

// opticksgeo-
#include "OpticksHub.hh"

// optixrap-
#include "OContext.hh"
#include "OpticksEntry.hh"
#include "OConfig.hh"
#include "OEvent.hh"
#include "OBuf.hh"
#include "OPropagator.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>
using namespace optix ; 


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


OPropagator::OPropagator(OpticksHub* hub, OEvent* oevt, OpticksEntry* entry) 
   :
    m_log(new SLog("OPropagator::OPropagator")),
    m_hub(hub),
    m_oevt(oevt),
    m_ocontext(m_oevt->getOContext()),
    m_ok(hub->getOpticks()),
    m_cfg(m_ok->getCfg()),
    m_override(m_cfg->getOverride()),
    m_entry(entry),
    m_entry_index(entry->getIndex()),
    m_prelaunch(false),
    m_rng_wrapper(NULL),
    m_count(0),
    m_width(0),
    m_height(0)
{
    init();
    (*m_log)("DONE");
}


void OPropagator::init()
{
    initParameters();
    initRng();
}


void OPropagator::initParameters()
{
    m_context = m_ocontext->getContext();

    m_context[ "propagate_epsilon"]->setFloat( m_ok->getEpsilon() );  // TODO: check impact of changing propagate_epsilon
    m_context[ "bounce_max" ]->setUint( m_ok->getBounceMax() );
    m_context[ "record_max" ]->setUint( m_ok->getRecordMax() );

    m_context[ "RNUMQUAD" ]->setUint( 2 );   // quads per record 
    m_context[ "PNUMQUAD" ]->setUint( 4 );   // quads per photon
    m_context[ "GNUMQUAD" ]->setUint( 6 );   // quads per genstep
    m_context["SPEED_OF_LIGHT"]->setFloat(299.792458f) ;   // mm/ns

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

    LOG(debug) << "OPropagator::initRng"
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




void OPropagator::setSize(unsigned width, unsigned height)
{
    m_width = width ; 
    m_height = height ; 
}

void OPropagator::prelaunch()
{
    assert(m_prelaunch == false);
    m_prelaunch = true ; 

    bool entry = m_entry_index > -1 ; 
    if(!entry) LOG(fatal) << "OPropagator::prelaunch MISSING entry " ;
    assert(entry);

    OpticksEvent* evt = m_oevt->getEvent(); 

    unsigned numPhotons = evt->getNumPhotons(); 
    setSize( m_override > 0 ? m_override : numPhotons ,  1 );

    STimes* prelaunch_times = evt->getPrelaunchTimes() ;

    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  0, 0, prelaunch_times ); 
    m_count += 1 ; 

    LOG(info) << prelaunch_times->description("prelaunch_times");
}

void OPropagator::launch()
{
    if(m_prelaunch == false) prelaunch();

    OpticksEvent* evt = m_oevt->getEvent(); 
    STimes* launch_times = evt->getLaunchTimes() ;

    m_ocontext->launch( OContext::LAUNCH,  m_entry_index,  m_width, m_height, launch_times);

    LOG(info) << launch_times->description("launch_times");
}

