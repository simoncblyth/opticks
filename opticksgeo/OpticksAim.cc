
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


const plog::Severity OpticksAim::LEVEL = debug ; 


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


    LOG(info)
        << " using CenterExtent from m_mesh0 "
        << " target " << target 
        << " aim " << aim
        << " ce " << gformat(ce) 
        << " for details : --aimdbg" 
        ;    

    m_composition->setCenterExtent(ce, aim); 
}

void OpticksAim::target()
{
    int target_ = getTarget() ;
    bool geocenter  = m_ok->hasOpt("geocenter");
    bool autocam = true ; 

    OpticksEvent* evt = m_hub->getEvent();

    if(target_ != 0)
    {
        LOG(info) << "OpticksAim::target SKIP as geometry target already set  " << target_ ; 
    }
    else if(geocenter )
    {
        glm::vec4 mmce = getCenterExtent();
        m_composition->setCenterExtent( mmce , autocam );
        LOG(info) << "OpticksAim::target (geocenter) mmce " << gformat(mmce) ; 
    }
    else if(evt && evt->hasGenstepData())
    {
        glm::vec4 gsce = evt->getGenstepCenterExtent();  // need to setGenStepData before this will work 
        m_composition->setCenterExtent( gsce , autocam );
        LOG(info) << "OpticksAim::target"
                  << " evt " << evt->brief()
                  << " gsce " << gformat(gsce) 
                  ; 
    }
}

