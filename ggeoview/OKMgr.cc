#include "OKMgr.hh"

class NConfigurable ; 

#include "SLog.hh"
#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#ifdef WITH_OPTIX
#include "OKPropagator.hh"  // ggeoview-
#endif

#define GUI_ 1
#include "OpticksViz.hh"

#include "PLOG.hh"
#include "GGV_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }


OKMgr::OKMgr(int argc, char** argv) 
    :
    m_log(new SLog("OKMgr::OKMgr")),
    m_ok(new Opticks(argc, argv, false)),   // false: NOT OKG4 integrated running
    m_hub(new OpticksHub(m_ok, true)),      // true: immediate configure and loadGeometry 
    m_idx(new OpticksIdx(m_hub)),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
#ifdef WITH_OPTIX
    m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
#endif
    m_count(0)
{
    init();
    (*m_log)("DONE");
}

void OKMgr::init()
{
    bool g4gun = m_ok->getSourceCode() == G4GUN ;
    if(g4gun)
         LOG(fatal) << "OKMgr doesnt support G4GUN, other that via loading (TO BE IMPLEMENTED) " ;
    assert(!g4gun);
}


void OKMgr::action()
{
    if(m_ok->hasOpt("load"))
    {
        loadPropagation();
    }
    else if(m_ok->hasOpt("nopropagate"))
    {
        LOG(info) << "--nopropagate/-P" ;
    }
    else
    { 
        int multi = m_ok->getMultiEvent();
        for(int i=0 ; i < multi ; i++) propagate();
    }
}

void OKMgr::propagate()
{
    m_count += 1 ; 
    LOG(fatal) << "OKMgr::propagate " << m_count ; 
    NPY<float>* gs = m_hub->getGensteps();  
#ifdef WITH_OPTIX
    m_propagator->propagate(gs);
#endif
    LOG(fatal) << "OKMgr::propagate DONE " << m_count ; 
}


void OKMgr::loadPropagation()
{
    LOG(fatal) << "OKMgr::loadPropagation" ; 

    m_hub->loadPersistedEvent(); 

    // formerly did indexing on load, 
    // but that is kinda crazy and should not normally be needed 
    // probably this was for G4 propagations prior to implementing
    // CPU indexing over in cfg4-

    if(!m_viz) return  ;

    m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

    m_viz->uploadEvent();  

    LOG(fatal) << "OKMgr::loadPropagation DONE" ; 
}

void OKMgr::visualize()
{
    if(m_viz) m_viz->visualize();
}

void OKMgr::cleanup()
{
#ifdef WITH_OPTIX
    m_propagator->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
}


