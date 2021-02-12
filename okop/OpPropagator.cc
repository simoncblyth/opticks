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

#include "NGLM.hpp"
#include "NPY.hpp"

#include "BTimeKeeper.hh"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#include "OpPropagator.hh" // okop-
#include "OpEngine.hh"  
#include "OpTracer.hh"  

#include "PLOG.hh"
#include "OKOP_BODY.hh"

const plog::Severity OpPropagator::LEVEL = PLOG::EnvLevel("OpPropagator", "DEBUG" ) ; 

OpPropagator::OpPropagator(OpticksHub* hub, OpticksIdx* idx) 
    :
    m_log(new SLog("OpPropagator::OpPropagator", "", LEVEL)),
    m_hub(hub),    
    m_idx(idx),
    m_ok(m_hub->getOpticks()),
    m_engine(new OpEngine(m_hub)),
    m_tracer(new OpTracer(m_engine,m_hub, true)),
    m_placeholder(0)
{
    (*m_log)("DONE");
}


/**
OpPropagator::propagate
-------------------------

Formerly this did not download when "m_ok->isSave()" was not active, 
but in production or nosave running with G4Opticks hit downloading is still 
needed even without opticks event saving being used.

**/

void OpPropagator::propagate()
{
    OK_PROFILE("_OpPropagator::propagate");

    OpticksEvent* evt = m_ok->getEvent();

    assert(evt);

    LOG(LEVEL) << "[ evtId(" << evt->getId() << ") " << m_ok->brief()   ;

    uploadEvent();

    m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries

    OK_PROFILE("OpPropagator::propagate");

    //int nhit = m_ok->isSave() ? downloadEvent() : -1 ; 
    int nhit = downloadEvent() ; 

    LOG(LEVEL) << "] evtId(" << evt->getId() << ") DONE nhit: " << nhit    ;

    OK_PROFILE("OpPropagator::propagate-download");
}

int OpPropagator::uploadEvent()
{
    LOG(LEVEL) << "[" ; 
    int npho = -1 ; 
    npho = m_engine->uploadEvent();
    LOG(LEVEL) << "]" ; 
    return npho ; 
}

int OpPropagator::downloadEvent()
{
    int nhit = -1 ; 
    nhit = m_engine->downloadEvent();
    return nhit ; 
}

void OpPropagator::indexEvent()
{
    m_idx->indexBoundariesHost();

    //m_idx->indexEvtOld();   // hostside checks, when saving makes sense 

    m_idx->indexSeqHost();
}

void OpPropagator::cleanup()
{
    m_engine->cleanup();
}

void OpPropagator::snap(const char* dir, const char* reldir)
{
    LOG(info) << " dir " << dir << " reldir " << reldir  ; 
    m_tracer->snap(dir, reldir);
}

