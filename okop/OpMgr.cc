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

#include "OpMgr.hh"

class NConfigurable ; 


#include "PLOG.hh"


#include "SLog.hh"
#include "BTimeKeeper.hh"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"

#include "OpticksHub.hh"    // okg-
#include "OpticksIdx.hh"    
#include "OpticksGen.hh"    
#include "OpticksGenstep.hh"    
#include "OpticksRun.hh"    

#include "OpEvt.hh"    

#include "OpPropagator.hh"  // okop-


const plog::Severity OpMgr::LEVEL = PLOG::EnvLevel("OpMgr", "DEBUG") ; 


// even though may end up doing the geocache check inside OpticksHub tis 
// convenient to have the Opticks instance outside OpMgr 

OpMgr::OpMgr(Opticks* ok ) 
    :
    m_log(new SLog("OpMgr::OpMgr","",LEVEL)),
    m_ok(ok ? ok : Opticks::GetInstance()),         
    m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry OR adopt a preexisting GGeo instance
    m_idx(new OpticksIdx(m_hub)),
    m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
    m_gen(m_hub->getGen()),
    m_run(m_hub->getRun()),
    m_propagator(new OpPropagator(m_hub, m_idx)),
    m_count(0)
{
    init();
    (*m_log)("DONE");
}

OpMgr::~OpMgr()
{
    cleanup();
}

void OpMgr::init()
{
    bool g4gun = m_ok->getSourceCode() == OpticksGenstep_G4GUN ;
    if(g4gun)
         LOG(fatal) << "OpMgr doesnt support G4GUN, other that via loading (TO BE IMPLEMENTED) " ;
    assert(!g4gun);

    //m_ok->dumpParameters("OpMgr::init");
}


void OpMgr::setGensteps(NPY<float>* gensteps)
{
    m_gensteps = gensteps ; 
}

//void OpMgr::uploadSensorLib(const SensorLib* sensorlib)
//{
//    m_propagator->uploadSensorLib(sensorlib);  
//}



OpticksEvent* OpMgr::getEvent() const 
{
    return m_run->getEvent() ; 
}
OpticksEvent* OpMgr::getG4Event() const 
{
    return m_run->getG4Event() ; 
}


/**
OpMgr::propagate
------------------

In "--production" mode post saving analysis is skipped.

NB this is exclusively used by G4Opticks, other than tests 

**/

void OpMgr::propagate()
{
    LOG(LEVEL) << "\n\n[[\n\n" ; 

    const Opticks& ok = *m_ok ; 
    
    if(ok("nopropagate")) return ; 

    assert( ok.isEmbedded() ); 

    assert( m_gensteps ); 

    bool production = m_ok->isProduction();

    bool compute = true ; 

    m_gensteps->setBufferSpec(OpticksEvent::GenstepSpec(compute));

    m_run->createEvent(m_gensteps); 

    m_propagator->propagate();

    if(ok("save")) 
    {
        LOG(LEVEL) << "( save " ;  
        m_run->saveEvent();
        LOG(LEVEL) << ") save " ;  

        LOG(LEVEL) << "( ana " ;  
        if(!production) m_hub->anaEvent();
        LOG(LEVEL) << ") ana " ;  
    }
    else
    {
        LOG(LEVEL) << "NOT saving " ;  
    }

    LOG(LEVEL) << "( postpropagate " ;  
    m_ok->postpropagate();  // profiling 
    LOG(LEVEL) << ") postpropagate " ;  

    LOG(LEVEL) << "\n\n]]\n\n" ; 
}


void OpMgr::reset()
{  
    LOG(LEVEL) << "[" ;  
    m_run->resetEvent();
    LOG(LEVEL) << "]" ;  
}

void OpMgr::cleanup()
{
   // m_propagator->cleanup();
    m_hub->cleanup();
    m_ok->cleanup(); 
}


/**
OpMgr::snap
-------------

Note that attempts to snap prior to an event upload fails with context invalid, 
see notes/issues/G4OKTest-snap-fails-with-invalid-context.rst

**/

void OpMgr::snap(const char* dir, const char* reldir)
{
    LOG(LEVEL) << "[" ; 
    m_propagator->snap(dir, reldir); 
    LOG(LEVEL) << "]" ; 
}


