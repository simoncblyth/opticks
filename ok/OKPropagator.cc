#include "SLog.hh"


#include "NGLM.hpp"
#include "NPY.hpp"

#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#include "OKPropagator.hh"

#ifdef WITH_OPTIX
#include "OKGLTracer.hh"     // optixgl-
#include "OpEngine.hh"  // opticksop-
#endif

#define GUI_ 1
#include "OpticksViz.hh"

#include "PLOG.hh"
#include "OK_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



OKPropagator::OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz) 
    :
    m_log(new SLog("OKPropagator::OKPropagator")),
    m_hub(hub),    
    m_idx(idx),
    m_viz(viz),    
    m_ok(m_hub->getOpticks()),
#ifdef WITH_OPTIX
    m_engine(new OpEngine(m_hub)),
    m_tracer(m_viz ? new OKGLTracer(m_engine,m_viz, true) : NULL ),
#endif
    m_placeholder(0)
{
    (*m_log)("DONE");
}




void OKPropagator::uploadEvent()
{
    if(m_viz) m_viz->uploadEvent();
#ifdef WITH_OPTIX
    m_engine->uploadEvent();
#endif
}


void OKPropagator::propagate()
{
    OK_PROFILE("OKPropagator::propagate.BEG");

    OpticksEvent* evt = m_hub->getEvent();

    assert(evt);

    LOG(fatal) << "OKPropagator::propagate(" << evt->getId() << ") " << m_ok->brief()   ;

    if(m_viz) m_hub->target();     // if not Scene targetted, point Camera at gensteps 

    uploadEvent();

    m_engine->propagate();        //  seedPhotonsFromGensteps, zeroRecords, propagate, indexSequence, indexBoundaries

    OK_PROFILE("OKPropagator::propagate.MID");

    if(m_viz) m_viz->indexPresentationPrep();

    bool trivial = m_ok->isTrivial();
 
    if(m_ok->hasOpt("save") || trivial) downloadEvent();

    if(trivial) trivialCheck();

    LOG(fatal) << "OKPropagator::propagate(" << evt->getId() << ") DONE "   ;

    OK_PROFILE("OKPropagator::propagate.END");
}


void OKPropagator::downloadEvent()
{
    if(m_viz) m_viz->downloadEvent();
#ifdef WITH_OPTIX
    m_engine->downloadEvent();
#endif
}



void OKPropagator::trivialCheck()
{
    LOG(fatal) << "OKPropagator::trivialCheck" ; 
  //  OpticksEvent* evt = m_hub->getEvent();
  //  NPY<float>* photon = evt->getPhotonData();
  //
  //  photon->dump("OKPropagator::trivialCheck");
  //  photon->save("$TMP/trivialCheck.npy");
}





void OKPropagator::indexEvent()
{
    m_idx->indexBoundariesHost();

    m_idx->indexEvtOld();   // hostside checks, when saving makes sense 

    m_idx->indexSeqHost();
}


void OKPropagator::cleanup()
{
#ifdef WITH_OPTIX
    m_engine->cleanup();
#endif
}




