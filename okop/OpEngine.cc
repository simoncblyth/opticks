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

#include "Opticks.hh"  // okc-
#include "OpticksEntry.hh" 
#include "OpticksHub.hh" // okg-
#include "OpticksSwitches.h" 
#include "SensorLib.hh"

// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
#include "OConfig.hh"
#include "OContext.hh"
#include "OEvent.hh"
#include "OPropagator.hh"
#include "OScene.hh"

#include "PLOG.hh"

const plog::Severity OpEngine::LEVEL = PLOG::EnvLevel("OpEngine", "DEBUG") ; 

unsigned OpEngine::getOptiXVersion() const 
{
   return OConfig::OptiXVersion();
}

OContext* OpEngine::getOContext() const 
{
    return m_scene->getOContext(); 
}

OPropagator* OpEngine::getOPropagator() const 
{
    return m_propagator ; 
}

int OpEngine::preinit() const
{
    LOG(LEVEL) ; 
    OKI_PROFILE("_OpEngine::OpEngine");
    return 0 ; 
}

OpEngine::OpEngine(OpticksHub* hub) 
    : 
    m_preinit(preinit()),
    m_log(new SLog("OpEngine::OpEngine","",LEVEL)),
    m_hub(hub),
    m_ok(m_hub->getOpticks()),
    m_scene(new OScene(m_hub)),
    m_ocontext(m_scene->getOContext()),
    m_entry(NULL),
    m_oevt(NULL),
    m_propagator(NULL),
    m_seeder(NULL),
    m_zeroer(NULL),
    m_indexer(NULL),
    m_closed(false)
{
    init();
    (*m_log)("DONE");
}

void OpEngine::init()
{
    LOG(LEVEL) << "[" ; 
    m_ok->setOptiXVersion(OConfig::OptiXVersion()); 

    bool is_load = m_ok->isLoad() ; 
    bool is_tracer = m_ok->isTracer() ;

    LOG(LEVEL) 
        << " is_load " << is_load 
        << " is_tracer " << is_tracer
        << " OptiXVersion " << m_ok->getOptiXVersion()
        ; 

    if(is_load)
    {
        LOG(LEVEL) << "skip initPropagation as just loading pre-cooked event " ;
    }
    else if(is_tracer)
    {
        LOG(LEVEL) << "skip initPropagation as tracer mode is active  " ; 
    }
    else
    {
        pLOG(LEVEL,0) << "(" ;  // -1 for one notch more logging 
        initPropagation(); 
        pLOG(LEVEL,0) << ")" ;
    }
    LOG(LEVEL) << "]" ; 
    OKI_PROFILE("OpEngine::OpEngine");
}

void OpEngine::uploadSensorLib(const SensorLib* sensorlib)
{
    m_scene->uploadSensorLib(sensorlib); 
}

/**
OpEngine::initPropagation
--------------------------

Instanciate the residents. Invoked by OpEngine::init

Note that the pointer to the single m_oevt (OEvent) instance  
is passed to all the residents.

**/

void OpEngine::initPropagation()
{
    LOG(LEVEL) << "[" ; 
    m_entry = m_ocontext->addEntry(m_ok->getEntryCode(), "OpEngine::initPropagation" ) ;
    LOG(LEVEL) << " entry " << m_entry->desc() ; 

    m_oevt = new OEvent(m_ok, m_ocontext);
    m_propagator = new OPropagator(m_ok, m_oevt, m_entry);
    m_seeder = new OpSeeder(m_ok, m_oevt) ;
    m_zeroer = new OpZeroer(m_ok, m_oevt) ;
    m_indexer = new OpIndexer(m_ok, m_oevt) ;
    LOG(LEVEL) << "]" ; 
}

/**
OpEngine::close
----------------

**/
void OpEngine::close()
{
    LOG(LEVEL) << "[" ; 
    assert( m_closed == false ); 
    m_closed = true ; 
    
    SensorLib* sensorlib = NULL ; 
    sensorlib = m_ok->getSensorLib(); 
    
    if( sensorlib == NULL )
    {
        LOG(info) << " sensorlib NULL : defaulting it with zero sensors " ; 
        unsigned num_sensors = 0 ; 
        m_ok->initSensorData(num_sensors);  
    }
    sensorlib = m_ok->getSensorLib(); 

    assert( sensorlib ); 
    sensorlib->close(); 
    assert( sensorlib->isClosed() ); 

    uploadSensorLib(sensorlib); 
    LOG(LEVEL) << "]" ; 
}

/**
OpEngine::uploadEvent
----------------------

With okop invoked from OpPropagator::propagate/OpPropagator::uploadEvent
With ok,okg4 invoked from OKPropagator::uploadEvent after OpticksViz::uploadEvent

**/

unsigned OpEngine::uploadEvent()
{
    LOG(LEVEL) << "[" ; 
    unsigned npho = m_oevt->upload();   // creates OptiX buffers, uploads gensteps
    LOG(LEVEL) << "] npho " << npho ; 
    return npho ; 
}

void OpEngine::propagate()
{
    LOG(LEVEL) << "[" ; 
    if(m_closed == false) close(); 

    LOG(LEVEL) << "( seeder.seedPhotonsFromGensteps ";  
    m_seeder->seedPhotonsFromGensteps();  // distributes genstep indices into the photons buffer OR seed buffer
    LOG(LEVEL) << ") seeder.seedPhotonsFromGensteps ";  

    m_oevt->markDirty();                   // inform OptiX that must sync with the CUDA modified photon/seed depending on WITH_SEED_BUFFER 

    //m_zeroer->zeroRecords();              // zeros on GPU record buffer via OptiX or OpenGL  (not working OptiX 4 in interop)

    LOG(LEVEL) << "( propagator.launch ";  
    m_propagator->launch();               // perform OptiX GPU propagation : write the photon, record and sequence buffers
    LOG(LEVEL) << ") propagator.launch ";  

    indexEvent();
    LOG(LEVEL) << "]" ; 
}


/**
OpEngine::indexEvent
---------------------

In production event indexing is skipped.


**/

void OpEngine::indexEvent()
{
    if(m_ok->isProduction()) return ; 

#ifdef WITH_RECORD
    m_indexer->indexSequence();
#endif
    m_indexer->indexBoundaries();
}


unsigned OpEngine::downloadEvent()
{
    LOG(LEVEL) << "[" ; 
    unsigned n = m_oevt->download();
    LOG(LEVEL) << "]" ; 
    return n ; 
}


void OpEngine::cleanup()
{
    m_scene->cleanup();
}

void OpEngine::Summary(const char* msg)
{
    LOG(info) << msg ; 
}


void OpEngine::downloadPhotonData()  // was used for debugging of seeding (buffer overwrite in interop mode on Linux)
{
     if(m_ok->isCompute()) m_oevt->downloadPhotonData(); 
}

