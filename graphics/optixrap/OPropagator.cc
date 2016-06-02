#include "OPropagator.hh"

// opticks-
#include "Opticks.hh"
#include "OpticksEvent.hh"

// optixrap-
#include "OContext.hh"
#include "OConfig.hh"
#include "OTimes.hh"
#include "OBuf.hh"
#include "OBufPair.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>

// npy-
#include "GLMPrint.hpp"
#include "timeutil.hpp"
#include "NPY.hpp"
#include "NLog.hpp"


using namespace optix ; 


// cudawrap-
#include "cuRANDWrapper.hh"


void OPropagator::init()
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
 
    const char* raygenprg = m_trivial ? "trivial" : "generate" ;  // last ditch debug technique
    LOG(debug) << "OPropagtor::init " << raygenprg ; 

    // OContext::e_generate_entry

    m_entry_index = m_ocontext->addRayGenerationProgram( "generate.cu.ptx", raygenprg );
    unsigned int exception_index = m_ocontext->addExceptionProgram( "generate.cu.ptx", "exception");
    assert(m_entry_index == exception_index);

    m_launch_times = new OTimes ; 
    m_prelaunch_times = new OTimes ; 

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

    unsigned int num_photons = m_evt->getNumPhotons();
    const char* rngCacheDir = OConfig::RngDir() ;

    LOG(info) << "OPropagator::initRng"
               << " rng_max " << rng_max
               << " num_photons " << num_photons 
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
    if(!m_evt) return ;
    initEvent(m_evt);
}

void OPropagator::initEvent(OpticksEvent* evt)  
{
    // when isInterop() == true 
    // the OptiX buffers for the evt data are actually references 
    // to the OpenGL buffers created with createBufferFromGLBO
    // by Scene::uploadEvt Scene::uploadSelection

    NPY<float>* gensteps =  evt->getGenstepData() ;

    m_genstep_buffer = m_ocontext->createIOBuffer<float>( gensteps, "gensteps");
    m_context["genstep_buffer"]->set( m_genstep_buffer );
    m_genstep_buf = new OBuf("genstep", m_genstep_buffer );

    if(m_ocontext->isCompute()) 
    {
        LOG(info) << "OPropagator::initGenerate (COMPUTE)" 
                  << " uploading gensteps "
                  ;
        OContext::upload<float>(m_genstep_buffer, gensteps);
    }
    else if(m_ocontext->isInterop())
    {
        assert(gensteps->getBufferId() > 0); 
        LOG(info) << "OPropagator::initGenerate (INTEROP)" 
                  << " gensteps handed to OptiX by referencing OpenGL buffer id  "
                  ;
    }

    m_photon_buffer = m_ocontext->createIOBuffer<float>( evt->getPhotonData(), "photon" );
    m_context["photon_buffer"]->set( m_photon_buffer );
    m_photon_buf = new OBuf("photon", m_photon_buffer );

    //if(evt->isStep())
    if(true)
    {
        NPY<short>* rx = evt->getRecordData() ;
        assert(rx);
        m_record_buffer = m_ocontext->createIOBuffer<short>( rx, "record");
        m_context["record_buffer"]->set( m_record_buffer );
        m_record_buf = new OBuf("record", m_record_buffer );


        NPY<unsigned long long>* sq = evt->getSequenceData() ;
        assert(sq);
        m_sequence_buffer = m_ocontext->createIOBuffer<unsigned long long>( sq, "sequence" );
        m_context["sequence_buffer"]->set( m_sequence_buffer );

        m_sequence_buf = new OBuf("sequence", m_sequence_buffer );
        m_sequence_buf->setMultiplicity(1u);
        m_sequence_buf->setHexDump(true);
    }
    else
    {
        LOG(info) << "OPropagator::initEvent skipping record/sequence buffer creation as nonStep " ;
    }


/*
//    m_aux_buffer = m_ocontext->createIOBuffer<short>( evt->getAuxData(), "aux" );
//    if(m_aux_buffer.get())
//       m_context["aux_buffer"]->set( m_aux_buffer );
*/


    // need to have done scene.uploadSelection for the recsel to have a buffer_id
}


void OPropagator::prelaunch()
{
    if(!m_evt) return ;

    unsigned int numPhotons = m_evt->getNumPhotons();

    bool enoughRng = numPhotons <= m_opticks->getRngMax() ;

    if(!enoughRng)
    {
        LOG(info) << "OPropagator::prelaunch"
                  << " not enoughRng "
                  << " numPhotons " << numPhotons 
                  << " rngMax " << m_opticks->getRngMax()
                  ;  
    } 

    assert( enoughRng  && "Use ggeoview-rng-prep to prepare RNG states up to the maximal number of photons generated " );

    m_width  = numPhotons ;
    m_height = 1 ;

    LOG(info) << "OPropagator::prelaunch count " << m_count << " size(" <<  m_width << "," <<  m_height << ")";

    if(m_override > 0)
    {
        m_width = m_override ; 
        LOG(warning) << "OPropagator::generate OVERRIDE photon count for debugging to " << m_width ; 
    }


    // OContext::e_generate_entry

    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  m_width, m_height, m_prelaunch_times);

    m_count += 1 ; 

    m_prelaunch = true ; 
}


void OPropagator::launch()
{
    assert(m_prelaunch && "must prelaunch before launch");

    // OContext::e_generate_entry

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
    if(!m_evt) return ;
    LOG(info)<<"OPropagator::downloadEvent" ;
 
    NPY<float>* dpho = m_evt->getPhotonData();
    OContext::download<float>( m_photon_buffer, dpho );

    //if(m_evt->isStep())
    if(true)
    {
        NPY<short>* drec = m_evt->getRecordData();
        OContext::download<short>( m_record_buffer, drec );

        NPY<unsigned long long>* dhis = m_evt->getSequenceData();
        OContext::download<unsigned long long>( m_sequence_buffer, dhis );
    }

    LOG(info)<<"OPropagator::downloadEvent DONE" ;
}

