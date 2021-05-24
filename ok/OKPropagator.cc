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

#include "OKPropagator.hh"

#ifdef OPTICKS_OPTIX
#include "OKGLTracer.hh"     // optixgl-
#include "OpEngine.hh"  // opticksop-
#endif

#define GUI_ 1
#include "OpticksViz.hh"

#include "PLOG.hh"
#include "OK_BODY.hh"


const plog::Severity OKPropagator::LEVEL = PLOG::EnvLevel("OKPropagator","DEBUG"); 


OKPropagator* OKPropagator::fInstance = NULL ; 
OKPropagator* OKPropagator::GetInstance(){ return fInstance ;}

int OKPropagator::preinit() const
{
    OKI_PROFILE("_OKPropagator::OKPropagator"); 
    return 0 ; 
}

OKPropagator::OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz) 
    :
    m_preinit(preinit()),
    m_log(new SLog("OKPropagator::OKPropagator", "", LEVEL)),
    m_hub(hub),    
    m_idx(idx),
    m_viz(viz),    
    m_ok(m_hub->getOpticks()),
#ifdef OPTICKS_OPTIX
    m_engine(new OpEngine(m_hub)),
    m_tracer(m_viz ? new OKGLTracer(m_engine,m_viz, true) : NULL ),
#endif
    m_placeholder(0)
{
    init(); 
}

void OKPropagator::init()
{
    (*m_log)("DONE");
    fInstance = this ; 
    OKI_PROFILE("OKPropagator::OKPropagator"); 
}

/**
OKPropagator::propagate
-------------------------

Used from OKMgr.

Question: 
   Why doesnt this use OpMgr like G4Opticks ? 
   Probably because OKMgr wants to work with graphics interop, 
   whereas OpMgr does not want to involve graphics.

**/

void OKPropagator::propagate()
{
    LOG(LEVEL) << "[" ; 
    OK_PROFILE("_OKPropagator::propagate");

    OpticksEvent* evt = m_ok->getEvent();

    assert(evt);

    LOG(LEVEL) << "OKPropagator::propagate(" << evt->getId() << ") " << m_ok->brief()   ;

    if(m_viz) m_hub->target();     // if not Scene targetted, point Camera at gensteps 

    uploadEvent();

    m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries

    OK_PROFILE("OKPropagator::propagate");

    if(m_viz) m_viz->indexPresentationPrep();

    int nhit = m_ok->isSave() ? downloadEvent() : -1 ; 

    LOG(LEVEL) << "OKPropagator::propagate(" << evt->getId() << ") DONE nhit: " << nhit    ;

    OK_PROFILE("OKPropagator::propagate-download");
    LOG(LEVEL) << "]" ; 
}

int OKPropagator::uploadEvent()
{
    char ctrl = '+' ; 

    if(m_viz) m_viz->uploadEvent(ctrl);

    int npho = -1 ; 
#ifdef OPTICKS_OPTIX
    npho = m_engine->uploadEvent();
#endif
    return npho ; 
}

int OKPropagator::downloadEvent()
{
    LOG(LEVEL) << "[" ; 
    if(m_viz) m_viz->downloadEvent();

    int nhit = -1 ; 
#ifdef OPTICKS_OPTIX
    nhit = m_engine->downloadEvent();
#endif
    LOG(LEVEL) << "]" ; 
    return nhit ; 
}

void OKPropagator::indexEvent()
{
    m_idx->indexBoundariesHost();

    //m_idx->indexEvtOld();   // hostside checks, when saving makes sense 

    m_idx->indexSeqHost();
}

void OKPropagator::cleanup()
{
#ifdef OPTICKS_OPTIX
    m_engine->cleanup();
#endif
}


