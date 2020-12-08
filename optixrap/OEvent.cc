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

#include "SLog.hh"
#include "NPY.hpp"
#include "NPho.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksPhoton.h"  
#include "OpticksFlags.hh"  
#include "OpticksEvent.hh"  
#include "OpticksBufferControl.hh"  

#include "OpticksHub.hh"    // okg-

#include "OContext.hh"
#include "OEvent.hh"
#include "OBuf.hh"

#include "CResource.hh"
#include "TBuf.hh"
// #include "cfloat4x4.h"   // fine with nvcc 


#include "PLOG.hh"


const plog::Severity OEvent::LEVEL = PLOG::EnvLevel("OEvent", "DEBUG") ; 


OpticksEvent* OEvent::getEvent() const 
{
    return m_evt ; 
}
void OEvent::setEvent(OpticksEvent* evt)
{
    LOG(LEVEL) << " this (OEvent*) " << this << " evt (OpticksEvent*) " << evt ; 
    m_evt = evt ; 
}
OContext* OEvent::getOContext() const 
{
    return m_ocontext ; 
}


OEvent::OEvent(Opticks* ok, OContext* ocontext)
    :
    m_log(new SLog("OEvent::OEvent", "", LEVEL)),
    m_ok(ok),
    m_hitmask(ok->getDbgHitMask()),  // default string from okc/OpticksCfg.cc "SD" converted to mask in Opticks::getDbgHitMask
    m_compute(ok->isCompute()),
    m_dbghit(m_ok->isDbgHit()),            // --dbghit
    m_dbgdownload(m_ok->isDbgDownload()),  // --dbgdownload
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
#ifdef WITH_DEBUG_BUFFER
    m_debug_buf(NULL),
#endif
#ifdef WITH_WAY_BUFFER
    m_way_buf(NULL),
#endif
    m_buffers_created(false)
{
    init();
    (*m_log)("DONE");
}


void OEvent::init()
{
    LOG(LEVEL)
        << " --dbghit " << ( m_dbghit ? "Y" : "N" )
        << " hitmask 0x" << std::hex << m_hitmask << std::dec
        << " " << OpticksFlags::FlagMask(m_hitmask, true)
        << " " << OpticksFlags::FlagMask(m_hitmask, false)
        ;
}



/**
OEvent::createBuffers
-----------------------

Invoked by OEvent::upload when buffers not yet created.

NB in INTEROP mode the OptiX buffers for the evt data 
are actually references to the OpenGL buffers created 
with createBufferFromGLBO by Scene::uploadEvt Scene::uploadSelection

HMM: want to do this in the init prior to real OpticksEvent/gensteps 
being available, like a static level action ? This would allow 
the context to become valid prior to uploading an event.


Interop with CUDA : Manual single-pointer synchronization (quote from OptiX 5.0 pdf)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a buffer’s contents are not changing for every launch, then the per-launch
copies of the automatic synchronization are not necessary. Automatic
synchronization can be disabled when creating a buffer by specifying the
RT_BUFFER_COPY_ON_DIRTY flag. With this flag, an application must call
rtBufferMarkDirty for synchronizations to take place. Calling rtBufferMarkDirty
after rtBufferUnmap will cause a synchronization from the buffer device pointer
at launch and override any pending synchronization from the host.

**/

void OEvent::createBuffers(OpticksEvent* evt)
{
    LOG(LEVEL) << evt->getShapeString() ; 
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

#ifdef WITH_DEBUG_BUFFER
    NPY<float>* debug_ = evt->getDebugData() ; 
    assert(debug_); 
    m_debug_buffer = m_ocontext->createBuffer<float>( debug_, "debug"); 
    m_context["debug_buffer"]->set( m_debug_buffer );
    m_debug_buf = new OBuf("debug", m_debug_buffer);
#endif

#ifdef WITH_WAY_BUFFER
    NPY<float>* way = evt->getWayData() ; 
    assert(way); 
    m_way_buffer = m_ocontext->createBuffer<float>( way, "way"); 
    m_context["way_buffer"]->set( m_way_buffer );
    m_way_buf = new OBuf("way", m_way_buffer);
#endif


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
    LOG(LEVEL); 

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


/**
OEvent::resizeBuffers
----------------------

Internally called by OEvent::upload

**/

void OEvent::resizeBuffers(OpticksEvent* evt)
{
    LOG(LEVEL) << evt->getShapeString() ; 

    NPY<float>* gensteps =  evt->getGenstepData() ;
    assert(gensteps);
    //OContext::resizeBuffer<float>(m_genstep_buffer, gensteps, "gensteps");
    m_ocontext->resizeBuffer<float>(m_genstep_buffer, gensteps, "gensteps");

    NPY<unsigned>* se = evt->getSeedData() ; 
    assert(se);
    m_ocontext->resizeBuffer<unsigned>(m_seed_buffer, se , "seed");

    NPY<float>* photon = evt->getPhotonData() ; 
    assert(photon);
    m_ocontext->resizeBuffer<float>(m_photon_buffer,  photon, "photon");

#ifdef WITH_SOURCE
    NPY<float>* source = evt->getSourceData() ; 
    if(source)
    {
        m_ocontext->resizeBuffer<float>(m_source_buffer,  source, "source");
    }
#endif


#ifdef WITH_RECORD
    NPY<short>* rx = evt->getRecordData() ; 
    assert(rx);
    m_ocontext->resizeBuffer<short>(m_record_buffer,  rx, "record");

    NPY<unsigned long long>* sq = evt->getSequenceData() ; 
    assert(sq);
    m_ocontext->resizeBuffer<unsigned long long>(m_sequence_buffer, sq , "sequence");
#endif
}


/**
OEvent::upload
----------------

Invoked by OpEngine::uploadEvent

**/

unsigned OEvent::upload()
{
    OpticksEvent* evt = m_ok->getEvent();
    assert(evt); 
    return upload(evt) ;  
}

unsigned OEvent::upload(OpticksEvent* evt)   
{
    OK_PROFILE("_OEvent::upload");
    LOG(LEVEL) << "[ id " << evt->getId()  ;
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

    LOG(LEVEL) << "] id " << evt->getId()  ;

    OK_PROFILE("OEvent::upload");

    return npho ;  
}


unsigned OEvent::uploadGensteps(OpticksEvent* evt)
{
    OK_PROFILE("_OEvent::uploadGensteps");
    NPY<float>* gensteps =  evt->getGenstepData() ;

    unsigned npho = evt->getNumPhotons();

    if(m_ocontext->isCompute()) 
    {
        LOG(LEVEL) << "(COMPUTE) id " << evt->getId() << " " << gensteps->getShapeString() << " -> " << npho  ;
        OContext::upload<float>(m_genstep_buffer, gensteps);
    }
    else if(m_ocontext->isInterop())
    {
        assert(gensteps->getBufferId() > 0); 
        LOG(LEVEL) << "(INTEROP) SKIP OpenGL BufferId " << gensteps->getBufferId()  ;
    }
    OK_PROFILE("OEvent::uploadGensteps");
    return npho ; 
}

unsigned OEvent::uploadSource(OpticksEvent* evt)
{
    OK_PROFILE("_OEvent::uploadSource");
    NPY<float>* source =  evt->getSourceData() ;
    if(!source) return 0 ; 

    unsigned nsrc = evt->getNumSource();

    if(m_ocontext->isCompute()) 
    {
        LOG(LEVEL) << "(COMPUTE) id " << evt->getId() << " " << source->getShapeString() << " -> " << nsrc  ;
        OContext::upload<float>(m_source_buffer, source);
    }
    else if(m_ocontext->isInterop())
    {
        assert(source->getBufferId() > 0); 
        LOG(LEVEL) << "(INTEROP) SKIP OpenGL BufferId " << source->getBufferId()  ;
    }
    OK_PROFILE("OEvent::uploadSource");
    return nsrc ; 
}





void OEvent::downloadPhotonData() 
{ 
    download(m_evt, PHOTON); 
}

unsigned OEvent::downloadHits()
{
    unsigned nhit = m_compute ? downloadHitsCompute(m_evt) : downloadHitsInterop(m_evt) ;

    LOG(info) 
        << " nhit " << nhit 
        << " --dbghit " << ( m_dbghit ? "Y" : "N" )
        << " hitmask 0x" << std::hex << m_hitmask << std::dec
        << " " << OpticksFlags::FlagMask(m_hitmask, true)
        << " " << OpticksFlags::FlagMask(m_hitmask, false)
        ;

    return nhit ; 
}


/**
OEvent::download
-------------------

In "--production" mode does not download the full event, only hits.

**/

unsigned OEvent::download()
{
    if(!m_ok->isProduction()) download(m_evt, DOWNLOAD_DEFAULT);

    unsigned nhit = downloadHits();  
    LOG(LEVEL) << " nhit " << nhit ; 

    return nhit ; 
}


/**
OEvent::download
-------------------

Note that hits are not downloaded with this, see OEvent::downloadHits 

**/

void OEvent::download(OpticksEvent* evt, unsigned mask)
{
    OK_PROFILE("_OEvent::download");
    assert(evt) ;

    LOG(LEVEL) << "[ id " << evt->getId()  ;
 
    if(mask & GENSTEP)
    {
        NPY<float>* gs = evt->getGenstepData();
        OContext::download<float>( m_genstep_buffer, gs );
        if(m_dbgdownload) LOG(info) << "gs " << gs->getShapeString() ;   
    }
    if(mask & SEED)
    {
        NPY<unsigned>* se = evt->getSeedData();
        OContext::download<unsigned>( m_seed_buffer, se );
        if(m_dbgdownload) LOG(info) << "se " << se->getShapeString() ;   
    }
    if(mask & PHOTON)   // final photon positions
    {
        NPY<float>* ox = evt->getPhotonData();
        OContext::download<float>( m_photon_buffer, ox );
        if(m_dbgdownload) LOG(info) << "ox " << ox->getShapeString() ;   
    }
#ifdef WITH_RECORD
    if(mask & RECORD)    // photon stepping point records 
    {
        NPY<short>* rx = evt->getRecordData();
        OContext::download<short>( m_record_buffer, rx );
        if(m_dbgdownload) LOG(info) << "rx " << rx->getShapeString() ;   
    }
    if(mask & SEQUENCE)   // seqhis, seqmat
    {
        NPY<unsigned long long>* sq = evt->getSequenceData();
        OContext::download<unsigned long long>( m_sequence_buffer, sq );
        if(m_dbgdownload) LOG(info) << "sq " << sq->getShapeString() ;   
    }
#endif

#ifdef WITH_DEBUG_BUFFER
    if(mask & DEBUG)  
    {
        NPY<float>* dg = evt->getDebugData();
        OContext::download<float>( m_debug_buffer, dg );
        if(m_dbgdownload) LOG(info) << "dg " << dg->getShapeString() ;   
    }
#endif

#ifdef WITH_WAY_BUFFER
    if(mask & WAY)  
    {
        NPY<float>* wy = evt->getWayData();
        OContext::download<float>( m_way_buffer, wy );
        if(m_dbgdownload) LOG(info) << "wy " << wy->getShapeString() ;   
    }
#endif


    LOG(LEVEL) << "]" ;
    OK_PROFILE("OEvent::download");
}



/**
OEvent::downloadHits
-------------------------

Downloading hits is special because a selection of the 
photon buffer is required, necessitating 
the use of Thrust stream compaction. This avoids allocating 
memory for all the photons on the host, just need to allocate
for the hits.

In interop need CUDA/Thrust access to underlying OpenGL buffer.  
In compute need CUDA/Thrust access to the OptiX buffer.  

**/

unsigned OEvent::downloadHitsCompute(OpticksEvent* evt)
{
    OK_PROFILE("_OEvent::downloadHitsCompute");

    NPY<float>* hit = evt->getHitData();

    CBufSpec cpho = m_photon_buf->bufspec();  
    assert( cpho.size % 4 == 0 );
    cpho.size /= 4 ;    //  decrease size by factor of 4, increases cpho "item" from 1*float4 to 4*float4 

    bool verbose = m_dbghit ; 
    TBuf tpho("tpho", cpho );
    unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, m_hitmask, verbose);
    // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    assert(hit->hasShape(nhit,4,4));

    OK_PROFILE("OEvent::downloadHitsCompute");

    LOG(LEVEL) 
         << " nhit " << nhit
         << " hit " << hit->getShapeString()
         ; 

    if(m_ok->isDumpHit())
    {
        unsigned maxDump = 100 ; 
        NPho::Dump(hit, maxDump, "OEvent::downloadHitsCompute --dumphit,post,flgs" ); 
    }

    return nhit ; 
}

unsigned OEvent::downloadHitsInterop(OpticksEvent* evt)
{
    OK_PROFILE("_OEvent::downloadHitsInterop");

    NPY<float>* hit = evt->getHitData();
    NPY<float>* npho = evt->getPhotonData();

    unsigned photon_id = npho->getBufferId();

    CResource rphoton( photon_id, CResource::R );
    //CBufSpec cpho = rphoton.mapGLToCUDA<cfloat4x4>();
    CBufSpec cpho = rphoton.mapGLToCUDA<float>();

    assert( cpho.size % 16 == 0 );
    cpho.size /= 16 ;    //  decrease size by factor of 16, increases cpho "item" from 1*float to 4*4*float 


    bool verbose = m_dbghit ; 
    TBuf tpho("tpho", cpho );
    unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, m_hitmask, verbose);
    // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    assert(hit->hasShape(nhit,4,4));


    OK_PROFILE("OEvent::downloadHitsInterop");
    LOG(LEVEL) 
         << " nhit " << nhit
         << " hit " << hit->getShapeString()
         ; 


    if(m_ok->isDumpHit())
    {
        unsigned maxDump = 100 ; 
        NPho::Dump(hit, maxDump, "OEvent::downloadHitsInterop --dumphit,post,flgs " ); 
    }

    return nhit ; 
}


OBuf* OEvent::getSeedBuf() const  {    return m_seed_buf ; } 
OBuf* OEvent::getPhotonBuf() const {   return m_photon_buf ; } 
OBuf* OEvent::getGenstepBuf() const {  return m_genstep_buf ; }
#ifdef WITH_RECORD
OBuf* OEvent::getRecordBuf() const  {  return m_record_buf ; } 
OBuf* OEvent::getSequenceBuf() const { return m_sequence_buf ; } 
#endif


