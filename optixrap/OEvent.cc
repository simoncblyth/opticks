#include "SLog.hh"
#include "NPY.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"  // okc-
#include "OpticksBufferControl.hh"  

#include "OpticksHub.hh"    // okg-

#include "OContext.hh"
#include "OEvent.hh"
#include "OBuf.hh"

#include "PLOG.hh"


OpticksEvent* OEvent::getEvent()
{
    return m_evt ; 
}
void OEvent::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 

}
OContext* OEvent::getOContext()
{
    return m_ocontext ; 
}


// canonical single OEvent instance resides in OPropagator 
// and is instanciated with OPropagator

OEvent::OEvent(OpticksHub* hub, OContext* ocontext)
   :
   m_log(new SLog("OEvent::OEvent")),
   m_hub(hub),
   m_ok(hub->getOpticks()),
   m_ocontext(ocontext),
   m_context(ocontext->getContext()),
   m_evt(NULL),
   m_photonMarkDirty(false),
   m_genstep_buf(NULL),
   m_photon_buf(NULL),
#ifdef WITH_RECORD
   m_record_buf(NULL),
   m_sequence_buf(NULL),
#endif
   m_buffers_created(false)
{
    (*m_log)("DONE");
}


void OEvent::createBuffers(OpticksEvent* evt)
{
    LOG(info) << "OEvent::createBuffers " << evt->getShapeString() ; 
    // NB in INTEROP mode the OptiX buffers for the evt data 
    // are actually references to the OpenGL buffers created 
    // with createBufferFromGLBO by Scene::uploadEvt Scene::uploadSelection

    assert(m_buffers_created==false);
    m_buffers_created = true ; 
 
    NPY<float>* gensteps =  evt->getGenstepData() ;
    assert(gensteps);
    m_genstep_buffer = m_ocontext->createBuffer<float>( gensteps, "gensteps");
    m_context["genstep_buffer"]->set( m_genstep_buffer );
    m_genstep_buf = new OBuf("genstep", m_genstep_buffer);

    NPY<unsigned>* se = evt->getSeedData() ;
    assert(se);
    m_seed_buffer = m_ocontext->createBuffer<unsigned>( se, "seed");  // name:seed triggers special case non-quad handling  
    m_context["seed_buffer"]->set( m_seed_buffer );
    m_seed_buf = new OBuf("seed", m_seed_buffer);
    m_seed_buf->setMultiplicity(1u);


    NPY<float>* photon = evt->getPhotonData() ; 
    assert(photon);

    OpticksBufferControl* photonCtrl = evt->getPhotonCtrl();
    m_photonMarkDirty = photonCtrl->isSet("BUFFER_COPY_ON_DIRTY") ;

    m_photon_buffer = m_ocontext->createBuffer<float>( photon, "photon");

    m_context["photon_buffer"]->set( m_photon_buffer );
    m_photon_buf = new OBuf("photon", m_photon_buffer);


#ifdef WITH_RECORD
    NPY<short>* rx = evt->getRecordData() ;
    assert(rx);
    m_record_buffer = m_ocontext->createBuffer<short>( rx, "record");
    m_context["record_buffer"]->set( m_record_buffer );
    m_record_buf = new OBuf("record", m_record_buffer);

    NPY<unsigned long long>* sq = evt->getSequenceData() ;
    assert(sq);
    m_sequence_buffer = m_ocontext->createBuffer<unsigned long long>( sq, "sequence"); 
    m_context["sequence_buffer"]->set( m_sequence_buffer );
    m_sequence_buf = new OBuf("sequence", m_sequence_buffer);
    m_sequence_buf->setMultiplicity(1u);
    m_sequence_buf->setHexDump(true);
#endif

}



void OEvent::markDirtyPhotonBuffer()
{
     if(m_photonMarkDirty)
     {
         LOG(info) << "OEvent::markDirtyPhotonBuffer PROCEED" ;
         m_photon_buffer->markDirty();   
     }
     else
     {
         LOG(info) << "OEvent::markDirtyPhotonBuffer SKIP " ;
     }

/*
2016-09-12 20:50:24.482 INFO  [438131] [OEvent::markDirtyPhotonBuffer@98] OEvent::markDirtyPhotonBuffer
libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtBufferMarkDirty(RTbuffer)" caught exception: Mark dirty only allowed on buffers created with RT_BUFFER_COPY_ON_DIRTY, file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Objects/Buffer.cpp, line: 867)
Abort trap: 6

2016-09-13 12:55:19.941 INFO  [495555] [OEvent::markDirtyPhotonBuffer@98] OEvent::markDirtyPhotonBuffer
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error 
            (Details: Function "RTresult _rtBufferMarkDirty(RTbuffer)" caught exception: 
             Must set or get buffer device pointer before calling rtBufferMarkDirty()., 
             file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Objects/Buffer.cpp, line: 861)
    Abort trap: 6

*/


}


void OEvent::resizeBuffers(OpticksEvent* evt)
{
    LOG(info) << "OEvent::resizeBuffers " << evt->getShapeString() ; 

    NPY<float>* gensteps =  evt->getGenstepData() ;
    assert(gensteps);
    OContext::resizeBuffer<float>(m_genstep_buffer, gensteps, "gensteps");

    NPY<unsigned>* se = evt->getSeedData() ; 
    assert(se);
    OContext::resizeBuffer<unsigned>(m_seed_buffer, se , "seed");

    NPY<float>* photon = evt->getPhotonData() ; 
    assert(photon);
    OContext::resizeBuffer<float>(m_photon_buffer,  photon, "photon");

#ifdef WITH_RECORD
    NPY<short>* rx = evt->getRecordData() ; 
    assert(rx);
    OContext::resizeBuffer<short>(m_record_buffer,  rx, "record");

    NPY<unsigned long long>* sq = evt->getSequenceData() ; 
    assert(sq);
    OContext::resizeBuffer<unsigned long long>(m_sequence_buffer, sq , "sequence");
#endif

}





void OEvent::upload()
{
    OpticksEvent* evt = m_hub->getEvent();
    assert(evt); 
    upload(evt) ;  
}

void OEvent::upload(OpticksEvent* evt)   
{
    LOG(info)<<"OEvent::upload id " << evt->getId()  ;
    setEvent(evt);

    if(!m_buffers_created)
    {
        createBuffers(evt);
    }
    else
    {
        resizeBuffers(evt);
    }
    uploadGensteps(evt);
    LOG(info)<<"OEvent::upload id " << evt->getId() << " DONE "  ;
}


void OEvent::uploadGensteps(OpticksEvent* evt)
{
    NPY<float>* gensteps =  evt->getGenstepData() ;
    if(m_ocontext->isCompute()) 
    {
        LOG(info) << "OEvent::uploadGensteps (COMPUTE)"  ;
        OContext::upload<float>(m_genstep_buffer, gensteps);
    }
    else if(m_ocontext->isInterop())
    {
        assert(gensteps->getBufferId() > 0); 
        LOG(info) << "OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId " << gensteps->getBufferId()  ;
    }
}



void OEvent::downloadPhotonData() { download(PHOTON); }
void OEvent::downloadSeedData()   { download(SEED); }
void OEvent::download(unsigned mask)
{
    download(m_evt, mask);
}


void OEvent::download(OpticksEvent* evt, unsigned mask)
{
    assert(evt) ;

   
    LOG(info)<<"OEvent::download id " << evt->getId()  ;
 
    if(mask & GENSTEP)
    {
        NPY<float>* genstep = evt->getGenstepData();
        OContext::download<float>( m_genstep_buffer, genstep );
    }
    if(mask & SEED)
    {
        NPY<unsigned>* se = evt->getSeedData();
        OContext::download<unsigned>( m_seed_buffer, se );
    }
    if(mask & PHOTON)
    {
       NPY<float>* photon = evt->getPhotonData();
       OContext::download<float>( m_photon_buffer, photon );
    }
#ifdef WITH_RECORD
    if(mask & RECORD)
    {
        NPY<short>* rx = evt->getRecordData();
        OContext::download<short>( m_record_buffer, rx );
    }
    if(mask & SEQUENCE)
    {
        NPY<unsigned long long>* sq = evt->getSequenceData();
        OContext::download<unsigned long long>( m_sequence_buffer, sq );
    }
#endif

    LOG(debug)<<"OEvent::download DONE" ;
}





OBuf* OEvent::getSeedBuf()
{
    return m_seed_buf ; 
}
OBuf* OEvent::getPhotonBuf()
{
    return m_photon_buf ; 
}
OBuf* OEvent::getGenstepBuf()
{
    return m_genstep_buf ; 
}


#ifdef WITH_RECORD
OBuf* OEvent::getRecordBuf()
{
    return m_record_buf ; 
}
OBuf* OEvent::getSequenceBuf()
{
    return m_sequence_buf ; 
}
#endif



