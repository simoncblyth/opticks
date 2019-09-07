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


#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 

#include "GMergedMesh.hh"
#include "GMergedMesh.hh"

#include "Composition.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksEvent.hh"
#include "OpticksAim.hh"

#include "PLOG.hh"


const plog::Severity OpticksAim::LEVEL = PLOG::EnvLevel("OpticksAim", "DEBUG") ; 


OpticksAim::OpticksAim(OpticksHub* hub) 
    :
    m_ok(hub->getOpticks()),
    m_dbgaim(m_ok->isDbgAim()),   // --dbgaim
    m_hub(hub),
    m_composition(hub->getComposition()),
    m_mesh0(NULL),
    m_target(0),
    m_target_deferred(0)
{
}


void OpticksAim::registerGeometry(GMergedMesh* mm0)
{
    m_mesh0 = mm0 ; 

    glm::vec4 ce0 = getCenterExtent(); 
    m_ok->setSpaceDomain( ce0 );

    LOG(m_dbgaim ? fatal : LEVEL)
          << " setting SpaceDomain : " 
          << " ce0 " << gformat(ce0) 
          ; 
}

glm::vec4 OpticksAim::getCenterExtent() 
{
    if(!m_mesh0)
    {
        LOG(fatal) << "OpticksAim::getCenterExtent" 
                   << " mesh0 NULL "
                   ;
        
        return glm::vec4(0.f,0.f,0.f,1.f) ;
    } 

    glm::vec4 mmce = GLMVEC4(m_mesh0->getCenterExtent(0)) ;
    return mmce ; 
}


void OpticksAim::dumpTarget(const char* msg)  
{
    m_hub->dumpVolumes( m_target, m_mesh0, msg  ); 
}


unsigned OpticksAim::getTargetDeferred()
{
    return m_target_deferred ;
}
unsigned OpticksAim::getTarget()
{
    return m_target ;
}




// TODO : consolidate the below 

void OpticksAim::setupCompositionTargetting()
{
    // used from OpticksViz::uploadGeometry

    //assert(0); 
    bool autocam = true ; 

    // handle commandline --target option that needs loaded geometry 
    unsigned deferred_target = getTargetDeferred();   // default to 0 
    unsigned cmdline_target = m_ok->getTarget();

    LOG(LEVEL)
        << " deferred_target " << deferred_target
        << " cmdline_target " << cmdline_target
        ;   

    setTarget(cmdline_target, autocam);
}

void  OpticksAim::setTarget(unsigned target, bool aim)  
{
    // formerly of oglrap-/Scene
    // invoked by OpticksViz::uploadGeometry OpticksViz::init

   if(m_mesh0 == NULL)
    {    
        LOG(info) << "OpticksAim::setTarget " << target << " deferring as geometry not loaded " ; 
        m_target_deferred = target ; 
        return ; 
    }    
    m_target = target ; 


    if(m_dbgaim)
    {
        dumpTarget("OpticksAim::setTarget");
    } 

    glm::vec4 ce = m_mesh0->getCE(target);


    LOG(LEVEL)
        << " using CenterExtent from m_mesh0 "
        << " target " << target 
        << " aim " << aim
        << " ce " << gformat(ce) 
        << " for details : --dbgaim " 
        ;    

    m_composition->setCenterExtent(ce, aim); 
}

void OpticksAim::target()
{
    int target_ = getTarget() ;
    bool geocenter  = m_ok->hasOpt("geocenter");  // --geocenter
    bool autocam = true ; 

    OpticksEvent* evt = m_hub->getEvent();

    if(target_ != 0)
    {
        LOG(info) << "SKIP as geometry target already set  " << target_ ; 
    }
    else if(geocenter )
    {
        glm::vec4 mmce = getCenterExtent();
        m_composition->setCenterExtent( mmce , autocam );
        LOG(LEVEL) << "[--geocenter] mmce " << gformat(mmce) ; 
    }
    else if(evt && evt->hasGenstepData())
    {
        glm::vec4 gsce = evt->getGenstepCenterExtent();  // need to setGenStepData before this will work 
        m_composition->setCenterExtent( gsce , autocam );
        LOG(LEVEL) 
            << " evt " << evt->brief()
            << " gsce " << gformat(gsce) 
            ; 
    }
}

