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


const plog::Severity OKPropagator::LEVEL = debug ; 


OKPropagator* OKPropagator::fInstance = NULL ; 
OKPropagator* OKPropagator::GetInstance(){ return fInstance ;}


OKPropagator::OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz) 
    :
    m_log(new SLog("OKPropagator::OKPropagator")),
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
    (*m_log)("DONE");
    fInstance = this ; 
}



void OKPropagator::propagate()
{
    OK_PROFILE("_OKPropagator::propagate");


    OpticksEvent* evt = m_hub->getEvent();

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
}



int OKPropagator::uploadEvent()
{
    if(m_viz) m_viz->uploadEvent();

    int npho = -1 ; 
#ifdef OPTICKS_OPTIX
    npho = m_engine->uploadEvent();
#endif
    return npho ; 
}

int OKPropagator::downloadEvent()
{
    if(m_viz) m_viz->downloadEvent();

    int nhit = -1 ; 
#ifdef OPTICKS_OPTIX
    nhit = m_engine->downloadEvent();
#endif
    return nhit ; 
}


void OKPropagator::indexEvent()
{
    m_idx->indexBoundariesHost();

    m_idx->indexEvtOld();   // hostside checks, when saving makes sense 

    m_idx->indexSeqHost();
}


void OKPropagator::cleanup()
{
#ifdef OPTICKS_OPTIX
    m_engine->cleanup();
#endif
}




