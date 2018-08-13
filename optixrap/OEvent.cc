#include "SLog.hh"
#include "NPY.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"  // okc-
#include "OpticksBufferControl.hh"  

#include "OpticksHub.hh"    // okg-

#include "OContext.hh"
#include "OEvent.hh"
#include "OBuf.hh"

#include "TBuf.hh"


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

OEvent::OEvent(Opticks* ok, OContext* ocontext)
   :
   m_log(new SLog("OEvent::OEvent")),
   m_ok(ok),
   m_mask(m_ok->getMaskBuffer()),
   m_ocontext(ocontext),
   m_context(ocontext->getContext()),
   m_evt(NULL),
   m_photonMarkDirty(false),
#ifdef WITH_SOURCE
   m_sourceMarkDirty(false),
#endif
   m_seedMarkDirty(false),
   m_genstep_buf(NULL),
   m_photon_buf(NULL),
#ifdef WITH_SOURCE
   m_source_buf(NULL),
#endif
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

    OpticksBufferControl* seedCtrl = evt->getSeedCtrl();
    m_seedMarkDirty = seedCtrl->isSet("BUFFER_COPY_ON_DIRTY") ;

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



#ifdef WITH_SOURCE
    NPY<float>* source = evt->getSourceData() ; 
    if(source)
    {
        OpticksBufferControl* sourceCtrl = evt->getSourceCtrl();
        m_sourceMarkDirty = sourceCtrl->isSet("BUFFER_COPY_ON_DIRTY") ;
        m_source_buffer = m_ocontext->createBuffer<float>( source, "source");
    } 
    else
    {
        m_source_buffer = m_ocontext->createEmptyBufferF4();
    }
    m_context["source_buffer"]->set( m_source_buffer );
    m_source_buf = new OBuf("source", m_source_buffer);
#endif




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



void OEvent::markDirty()
{

#ifdef WITH_SEED_BUFFER

     if(m_seedMarkDirty)
     {
         LOG(info) << "OEvent::markDirty(seed) PROCEED" ;
         m_seed_buffer->markDirty();   
     }
     else
     {
         LOG(debug) << "OEvent::markDirty(seed) SKIP " ;
     }

#else

     if(m_photonMarkDirty)
     {
         LOG(info) << "OEvent::markDirty(photon) PROCEED" ;
         m_photon_buffer->markDirty();   
     }
     else
     {
         LOG(debug) << "OEvent::markDirty(photon) SKIP " ;
     }

#endif



#ifdef WITH_SOURCE
     if(m_sourceMarkDirty)
     {
         LOG(info) << "OEvent::markDirty(source) PROCEED" ;
         m_source_buffer->markDirty();   
     }
     else
     {
         LOG(debug) << "OEvent::markDirty(source) SKIP " ;
     }
#endif



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
    LOG(debug) << "OEvent::resizeBuffers " << evt->getShapeString() ; 

    NPY<float>* gensteps =  evt->getGenstepData() ;
    assert(gensteps);
    OContext::resizeBuffer<float>(m_genstep_buffer, gensteps, "gensteps");

    NPY<unsigned>* se = evt->getSeedData() ; 
    assert(se);
    OContext::resizeBuffer<unsigned>(m_seed_buffer, se , "seed");

    NPY<float>* photon = evt->getPhotonData() ; 
    assert(photon);
    OContext::resizeBuffer<float>(m_photon_buffer,  photon, "photon");

#ifdef WITH_SOURCE
    NPY<float>* source = evt->getSourceData() ; 
    if(source)
    {
        OContext::resizeBuffer<float>(m_source_buffer,  source, "source");
    }
#endif


#ifdef WITH_RECORD
    NPY<short>* rx = evt->getRecordData() ; 
    assert(rx);
    OContext::resizeBuffer<short>(m_record_buffer,  rx, "record");

    NPY<unsigned long long>* sq = evt->getSequenceData() ; 
    assert(sq);
    OContext::resizeBuffer<unsigned long long>(m_sequence_buffer, sq , "sequence");
#endif
}



unsigned OEvent::upload()
{
    OpticksEvent* evt = m_ok->getEvent();
    assert(evt); 
    return upload(evt) ;  
}

unsigned OEvent::upload(OpticksEvent* evt)   
{
    OK_PROFILE("_OEvent::upload");
    LOG(debug)<<"OEvent::upload id " << evt->getId()  ;
    setEvent(evt);

    if(!m_buffers_created)
    {
        createBuffers(evt);
    }
    else
    {
        resizeBuffers(evt);
    }
    unsigned npho = uploadGensteps(evt);
    unsigned nsrc = uploadSource(evt);

    if( nsrc > 0 )
    {
        assert( nsrc == npho ); 
    }

    LOG(debug)<<"OEvent::upload id " << evt->getId() << " DONE "  ;

    OK_PROFILE("OEvent::upload");

    return npho ;  
}


unsigned OEvent::uploadGensteps(OpticksEvent* evt)
{
    NPY<float>* gensteps =  evt->getGenstepData() ;

    unsigned npho = evt->getNumPhotons();

    if(m_ocontext->isCompute()) 
    {
        LOG(info) << "OEvent::uploadGensteps (COMPUTE) id " << evt->getId() << " " << gensteps->getShapeString() << " -> " << npho  ;
        OContext::upload<float>(m_genstep_buffer, gensteps);
    }
    else if(m_ocontext->isInterop())
    {
        assert(gensteps->getBufferId() > 0); 
        LOG(info) << "OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId " << gensteps->getBufferId()  ;
    }
    return npho ; 
}

unsigned OEvent::uploadSource(OpticksEvent* evt)
{
    NPY<float>* source =  evt->getSourceData() ;
    if(!source) return 0 ; 

    unsigned nsrc = evt->getNumSource();

    if(m_ocontext->isCompute()) 
    {
        LOG(info) << "OEvent::uploadSource (COMPUTE) id " << evt->getId() << " " << source->getShapeString() << " -> " << nsrc  ;
        OContext::upload<float>(m_source_buffer, source);
    }
    else if(m_ocontext->isInterop())
    {
        assert(source->getBufferId() > 0); 
        LOG(info) << "OEvent::uploadSource (INTEROP) SKIP OpenGL BufferId " << source->getBufferId()  ;
    }
    return nsrc ; 

}





void OEvent::downloadPhotonData() 
{ 
    download(m_evt, PHOTON); 
}

unsigned OEvent::downloadHits()
{
    return downloadHits(m_evt);
}

unsigned OEvent::download()
{
    if(!m_ok->isProduction()) download(m_evt, DOWNLOAD_DEFAULT);
    return downloadHits(m_evt);  
}


/** OEvent::download

 
**/


void OEvent::download(OpticksEvent* evt, unsigned mask)
{
    OK_PROFILE("_OEvent::download");
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
    OK_PROFILE("OEvent::download");
}


unsigned OEvent::downloadHits(OpticksEvent* evt)
{
    OK_PROFILE("_OEvent::downloadHits");

    NPY<float>* hit = evt->getHitData();

    LOG(error) << "OEvent::downloadHits.cpho" ;
    CBufSpec cpho = m_photon_buf->bufspec();  
    LOG(error) << "OEvent::downloadHits.cpho DONE " ;
    assert( cpho.size % 4 == 0 );
    cpho.size /= 4 ;    //  decrease size by factor of 4, increases cpho "item" from 1*float4 to 4*float4 

    bool verbose = false ; 
    TBuf tpho("tpho", cpho );
    unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, verbose);
    // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    assert(hit->hasShape(nhit,4,4));

    OK_PROFILE("OEvent::downloadHits");

    return nhit ; 
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



