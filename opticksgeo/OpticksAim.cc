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

//#define GLMVEC4(g) glm::vec4((g).x,(g).y,(g).z,(g).w) 

#include "GGeo.hh"

#include "Composition.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksEvent.hh"
#include "OpticksAim.hh"

#include "PLOG.hh"


const plog::Severity OpticksAim::LEVEL = PLOG::EnvLevel("OpticksAim", "DEBUG") ; 

int OpticksAim::Preinit() // static 
{
    LOG(LEVEL);
    return 0 ; 
}

OpticksAim::OpticksAim(OpticksHub* hub) 
    :
    m_preinit(Preinit()), 
    m_ok(hub->getOpticks()),
    m_dbgaim(m_ok->isDbgAim()),   // --dbgaim
    m_hub(hub),
    m_composition(hub->getComposition()),
    m_ggeo(NULL),
    m_target(0),
    m_gdmlaux_target(0),
    m_cmdline_targetpvn(0),
    m_autocam(true)
{
    init(); 
}

void OpticksAim::init()
{
    LOG(LEVEL); 
}


/**
OpticksAim::registerGeometry
------------------------------

Canonically invoked by OpticksHub::loadGeometry OR OpticksHub::adoptGeometry

**/

void OpticksAim::registerGeometry(GGeo* ggeo)
{
    assert( ggeo ); 
    m_ggeo = ggeo ; 

    const char* gdmlaux_target_lvname = m_ok->getGDMLAuxTargetLVName() ; 
    m_gdmlaux_target =  m_ggeo->getFirstNodeIndexForGDMLAuxTargetLVName() ; // sensitive to GDML auxilary lvname metadata (label, target)  

    const char* targetpvn = m_ok->getTargetPVN();     // --targetpvn
    m_cmdline_targetpvn = m_ggeo->getFirstNodeIndexForPVNameStarting(targetpvn) ;   

    int cmdline_domaintarget = m_ok->getDomainTarget();    // --domaintarget 

    unsigned active_domaintarget = 0 ;  
    if( cmdline_domaintarget > 0 )
    {
        active_domaintarget = cmdline_domaintarget ; 
    } 
    else if( m_gdmlaux_target > 0 )
    {
        active_domaintarget = m_gdmlaux_target ; 
    }

    m_targets["gdmlaux_domain"] = m_gdmlaux_target  ; 
    m_targets["cmdline_domain"] = cmdline_domaintarget  ; 
    m_targets["cmdline_pvn"] = m_cmdline_targetpvn  ; 
    m_targets["active_domain"] = active_domaintarget  ; 

    LOG(LEVEL)
        << " cmdline_domaintarget [--domaintarget] " << cmdline_domaintarget
        << " gdmlaux_target " << m_gdmlaux_target
        << " gdmlaux_target_lvname  " << gdmlaux_target_lvname 
        << " active_domaintarget " << active_domaintarget
        << " targetpvn " << targetpvn 
        << " cmdline_targetpvn " << m_cmdline_targetpvn  
        ; 

    glm::vec4 center_extent = m_ggeo->getCE(active_domaintarget); 

    LOG(LEVEL)
        << " setting SpaceDomain : " 
        << " active_domaintarget " << active_domaintarget
        << " center_extent " << gformat(center_extent) 
        ; 
    
    m_ok->setSpaceDomain( center_extent );
}

void OpticksAim::dumpTarget(const char* msg) const 
{
    assert( m_ggeo ); 
    float extent_cut_mm = 5000.f ; 
    m_ggeo->dumpVolumes(m_targets, msg, extent_cut_mm, m_target ); 
}


unsigned OpticksAim::getTarget() const 
{
    return m_target ;
}


/**
OpticksAim::setupCompositionTargetting
-----------------------------------------

Relayed via OpticksHub::setupCompositionTargetting from eg OpticksViz::uploadGeometry or OpTracer::render

Decides on the target volume node index to configure for rendering. 
Priority order of inputs to control the target volume:

1. command line "--target 3155" option, with target defaulting to the value of OPTICKS_TARGET envvar or fallback 0  
2. deferred target if a request was made prior to geometry being loaded (is this still needed?)
3. GDMLAux metadata that annotates a logical volume via GDML auxiliary elements with (key,value) ("label","target"),
   the node index of the first physical placed volume instance of that logical volume

**/

void OpticksAim::setupCompositionTargetting()
{
    unsigned cmdline_target = m_ok->getTarget();      // sensitive to OPTICKS_TARGET envvar, fallback 0 
    unsigned active_target = 0 ; 

    if( m_cmdline_targetpvn > 0 ) 
    {
        active_target = m_cmdline_targetpvn ; 
    } 
    else if( cmdline_target > 0 )
    {
        active_target = cmdline_target ; 
    } 
    else if( m_gdmlaux_target > 0 )
    {
        active_target = m_gdmlaux_target ;
    }

    m_targets["gdmlaux_composition"] = m_gdmlaux_target ; 
    m_targets["cmdline_composition"] = cmdline_target ; 
    m_targets["active_composition" ] = active_target ; 

    LOG(error)
        << " cmdline_targetpvn " << m_cmdline_targetpvn
        << " cmdline_target " << cmdline_target
        << " gdmlaux_target " << m_gdmlaux_target  
        << " active_target " << active_target 
        ;   

    setTarget(active_target, m_autocam);
}

/**
OpticksAim::setTarget
----------------------

Sets the composition center_extent to that of the target 
volume identified by node index.

If this is invoked prior to registering geometry the 
target is retained in m_target_deferred.
   
Invoked by OpticksViz::uploadGeometry OpticksViz::init OpticksHub::setTarget

Formerly of oglrap-/Scene

**/

void  OpticksAim::setTarget(unsigned target, bool aim)  
{
    assert(m_ggeo) ;  // surely always now available ?

    m_target = target ; 
    if(m_dbgaim) dumpTarget("OpticksAim::setTarget"); 

    glm::vec4 ce = m_ggeo->getCE(target);
    LOG(LEVEL)
        << " using CenterExtent from m_ggeo "
        << " target " << target 
        << " aim " << aim
        << " ce " << gformat(ce) 
        << " for details : --dbgaim " 
        ;    

    m_composition->setCenterExtent(ce, aim); 
}


/**
OpticksAim::target
------------------

Sets composition center_extent depending on the target, 
presence of commandline option --geocenter and event gensteps.
This controls the view position and orientation used by rendering.

**/

void OpticksAim::target()
{
    int target_ = getTarget() ;
    bool geocenter  = m_ok->hasOpt("geocenter");  // --geocenter

    OpticksEvent* evt = m_ok->getEvent();

    if(target_ != 0)
    {
        LOG(LEVEL) << "SKIP as geometry target already set  " << target_ ; 
    }
    else if(geocenter )
    {
        glm::vec4 ce0 = m_ggeo->getCE(0);
        m_composition->setCenterExtent( ce0 , m_autocam );
        LOG(LEVEL) << "[--geocenter] ce0 " << gformat(ce0) ; 
    }
    else if(evt && evt->hasGenstepData())
    {
        glm::vec4 gsce = evt->getGenstepCenterExtent();  // need to setGenStepData before this will work 
        m_composition->setCenterExtent( gsce , m_autocam );
        LOG(LEVEL) 
            << " evt " << evt->brief()
            << " gsce " << gformat(gsce) 
            ; 
    }
}


