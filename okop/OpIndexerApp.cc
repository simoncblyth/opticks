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

#include <cstddef>


// opticks-
#include "OpticksSwitches.h"
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "OpticksEvent.hh"

// opticksgeo-
#include "OpticksHub.hh"
#include "OpticksRun.hh"

// optixrap-
#include "OEvent.hh"
#include "OScene.hh"

// opop-
#include "OpIndexerApp.hh"
#include "OpIndexer.hh"

// npy-
#include "PLOG.hh"



OpIndexerApp::OpIndexerApp(int argc, char** argv) 
   :   
     m_ok(new Opticks(argc, argv)),
     m_cfg(m_ok->getCfg()),
     m_hub(new OpticksHub(m_ok)),
     m_run(m_ok->getRun()),
     m_scene(new OScene(m_hub)),
     m_ocontext(m_scene->getOContext()),
     m_oevt(new OEvent(m_ok,m_ocontext)),
     m_indexer(new OpIndexer(m_ok, m_oevt))
{
}

void OpIndexerApp::loadEvtFromFile()
{
    m_ok->setSpaceDomain(0.f,0.f,0.f,1000.f);  // this is required before can create an evt 

    m_run->loadEvent();

    OpticksEvent* evt = m_run->getEvent();
    evt->Summary("OpIndexerApp::configure");
 
    if(evt->isNoLoad())
    {    
        LOG(info) << "App::loadEvtFromFile LOAD FAILED " ;
        return ; 
    }    

}

void OpIndexerApp::makeIndex()
{
    OpticksEvent* evt = m_run->getEvent();
    if(evt->isIndexed())
    {
        bool forceindex = m_ok->hasOpt("forceindex");
        if(forceindex)
        {
            LOG(info) << "OpIndexerApp::makeIndex evt is indexed already, but --forceindex option in use, so proceeding..." ;
        }
        else
        {
            LOG(info) << "OpIndexerApp::makeIndex evt is indexed already, SKIPPING " ;
            return  ;
        }
    }
    if(evt->isNoLoad())
    {
        LOG(info) << "OpIndexerApp::makeIndex evt failed to load, SKIPPING " ;
        return  ;
    }


    evt->Summary("OpIndexerApp::makeIndex");

    //evt->prepareForIndexing();
#ifdef WITH_RECORD
    m_indexer->indexSequence();
#else
    LOG(fatal) << " compile WITH_RECORD to indexSequence " ;
#endif

    evt->saveIndex();
}




