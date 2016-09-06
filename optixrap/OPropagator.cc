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
#include "OTimes.hh"

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
    return m_sequence_buf ; 
}
OBuf* OPropagator::getPhotonBuf()
{
    return m_photon_buf ; 
}
OBuf* OPropagator::getGenstepBuf()
{
    return m_genstep_buf ; 
}
OBuf* OPropagator::getRecordBuf()
{
    return m_record_buf ; 
}
OTimes* OPropagator::getPrelaunchTimes()
{
    return m_prelaunch_times ; 
}
OTimes* OPropagator::getLaunchTimes()
{
    return m_launch_times ; 
}




OPropagator::OPropagator(OContext* ocontext, OpticksHub* hub, int override_) 
   :
    m_ocontext(ocontext),
    m_hub(hub),
    m_opticks(hub->getOpticks()),
    m_prelaunch_times(new OTimes),
    m_launch_times(new OTimes),
    m_prelaunch(false),
    m_entry_index(-1),
    m_photon_buf(NULL),
    m_sequence_buf(NULL),
    m_genstep_buf(NULL),
    m_record_buf(NULL),
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

}




void OPropagator::initParameters()
{
    m_context = m_ocontext->getContext();

    m_context[ "propagate_epsilon"]->setFloat( m_opticks->getEpsilon() );  // TODO: check impact of changing propagate_epsilon
    m_context[ "bounce_max" ]->setUint( m_opticks->getBounceMax() );
    m_context[ "record_max" ]->setUint( m_opticks->getRecordMax() );

    optix::uint4 debugControl = optix::make_uint4(m_ocontext->getDebugPhoton(),0,0,0);
    LOG(debug) << "OPropagator::init debugControl " 
              << " x " << debugControl.x 
              << " y " << debugControl.y
              << " z " << debugControl.z 
              << " w " << debugControl.w 
              ;

    m_context["debug_control"]->setUint(debugControl); 
 
    const glm::vec4& ce = m_opticks->getSpaceDomain();
    const glm::vec4& td = m_opticks->getTimeDomain();

    m_context["center_extent"]->setFloat( make_float4( ce.x, ce.y, ce.z, ce.w ));
    m_context["time_domain"]->setFloat(   make_float4( td.x, td.y, td.z, td.w ));
}






void OPropagator::initRng()
{
    unsigned int rng_max = m_opticks->getRngMax();
    if(rng_max == 0 )
    {
        LOG(warning) << "OPropagator::initRng"   
                     << " EARLY EXIT "
                     << " rng_max " << rng_max
                     ;
        return ;
    }

    // hmm evt should not be needed at init stage, 

    //OpticksEvent* evt = m_hub->getEvent();
    //unsigned int num_photons = evt->getNumPhotons();

    const char* rngCacheDir = m_opticks->getRNGInstallCacheDir();

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

    bool enoughRng = numPhotons <= m_opticks->getRngMax() ;

    if(!enoughRng)
    {
        LOG(info) << "OPropagator::initEvent"
                  << " not enoughRng "
                  << " numPhotons " << numPhotons 
                  << " rngMax " << m_opticks->getRngMax()
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


    initEvent(evt);
}




// creates OBuf for gensteps, photon, record, sequence and uploads gensteps in compute, already there in interop
void OPropagator::initEvent(OpticksEvent* evt)   
{
    // when isInterop() == true 
    // the OptiX buffers for the evt data are actually references 
    // to the OpenGL buffers created with createBufferFromGLBO
    // by Scene::uploadEvt Scene::uploadSelection
    // 
    //
    // Hear are recreating buffer for each evt, 
    // could try reusing OBuf ?


    LOG(info) << "OPropagator::initEvent" ; 

    NPY<float>* gensteps =  evt->getGenstepData() ;

    m_genstep_buffer = m_ocontext->createBuffer<float>( gensteps, "gensteps");
    m_context["genstep_buffer"]->set( m_genstep_buffer );
    m_genstep_buf = new OBuf("genstep", m_genstep_buffer, gensteps);

    if(m_ocontext->isCompute()) 
    {
        LOG(info) << "OPropagator::initGenerate (COMPUTE)" << " uploading gensteps " ;
        OContext::upload<float>(m_genstep_buffer, gensteps);
    }
    else if(m_ocontext->isInterop())
    {
        assert(gensteps->getBufferId() > 0); 
        LOG(info) << "OPropagator::initGenerate (INTEROP)" 
                  << " gensteps handed to OptiX by referencing OpenGL buffer id  "
                  ;
    }


    NPY<float>* photon = evt->getPhotonData() ; 
    m_photon_buffer = m_ocontext->createBuffer<float>( photon, "photon");
    m_context["photon_buffer"]->set( m_photon_buffer );
    m_photon_buf = new OBuf("photon", m_photon_buffer, photon);

    // photon buffer is OPTIX_INPUT_OUTPUT (INPUT for genstep seeds) 
    // but the seeding is done GPU side via thrust  


    if(m_opticks->hasOpt("dbginterop"))
    {
        LOG(info) << "OPropagator::initEvent skipping record/sequence buffer creation as dbginterop  " ;
    }
    else
    {
        NPY<short>* rx = evt->getRecordData() ;
        assert(rx);
        m_record_buffer = m_ocontext->createBuffer<short>( rx, "record");
        m_context["record_buffer"]->set( m_record_buffer );
        m_record_buf = new OBuf("record", m_record_buffer, rx);



        NPY<unsigned long long>* sq = evt->getSequenceData() ;
        assert(sq);

        m_sequence_buffer = m_ocontext->createBuffer<unsigned long long>( sq, "sequence"); 
        m_context["sequence_buffer"]->set( m_sequence_buffer );
        m_sequence_buf = new OBuf("sequence", m_sequence_buffer, sq);
        m_sequence_buf->setMultiplicity(1u);
        m_sequence_buf->setHexDump(true);

        // sequence buffer requirements:
        //
        //     * written by OptiX
        //     * read by CUDA/Thrust in order to create the index
        //     * not-touched by OpenGL, OpenGL only needs access to the index buffer written by Thrust 
        //

    }
    LOG(info) << "OPropagator::initEvent DONE" ; 
}


void OPropagator::prelaunch()
{
    bool entry = m_entry_index > -1 ; 
    if(!entry)
    {
        LOG(fatal) << "OPropagator::prelaunch"
                   << " must setEntry before prelaunch/launch  "
                  ;  
 
    }
    assert(entry);

    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  0, 0, m_prelaunch_times); 

    // "prelaunch" by definition needs no dimensions, so it can be done prior to having an event 
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



void OPropagator::downloadPhotonData()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ;
    LOG(info)<<"OPropagator::downloadPhotonData" ;
 
    NPY<float>* dpho = evt->getPhotonData();
    OContext::download<float>( m_photon_buffer, dpho );
}


void OPropagator::downloadEvent()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ;
    LOG(info)<<"OPropagator::downloadEvent" ;
 

    NPY<float>* dpho = evt->getPhotonData();
    OContext::download<float>( m_photon_buffer, dpho );

    NPY<short>* drec = evt->getRecordData();
    OContext::download<short>( m_record_buffer, drec );

    NPY<unsigned long long>* dhis = evt->getSequenceData();
    OContext::download<unsigned long long>( m_sequence_buffer, dhis );


    LOG(info)<<"OPropagator::downloadEvent DONE" ;
}

