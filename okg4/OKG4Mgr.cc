#include "OKG4Mgr.hh"

class NConfigurable ; 

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

#include "CG4.hh"
#include "CCollector.hh"

#include "PLOG.hh"
#include "OKG4_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }

OKG4Mgr::OKG4Mgr(int argc, char** argv) 
    :
    m_ok(new Opticks(argc, argv, true)),               // true: integrated running 
    m_hub(new OpticksHub(m_ok, true)),                 // true: configure, loadGeometry and setupInputGensteps immediately
    m_idx(new OpticksIdx(m_hub)),
    m_g4(new CG4(m_hub, true)),                        // true: configure and initialize immediately 
    m_collector(new CCollector(m_hub->getLookup())),   // after CG4 loads geometry, for material code cross-referenceing in NLookup
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
#ifdef WITH_OPTIX
    m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
#endif
    m_placeholder(0)
{
    init();
    LOG(fatal) << "OKG4Mgr::OKG4Mgr DONE" ;  
}

void OKG4Mgr::init()
{
}

void OKG4Mgr::propagate()
{
    LOG(fatal) << "OKG4Mgr::propagate" ;

    m_g4->propagate();

    NPY<float>* gs = m_ok->isLiveGensteps() ? m_collector->getGensteps() : m_hub->getGensteps() ;  

    // collected from G4 directly OR input gensteps fabricated from config (eg torch) or loaded from file

    m_propagator->propagate(gs);

    LOG(fatal) << "OKG4Mgr::propagate DONE" ;
}

void OKG4Mgr::visualize()
{
    if(m_viz) m_viz->visualize();
}

void OKG4Mgr::cleanup()
{
#ifdef WITH_OPTIX
    m_propagator->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
    m_g4->cleanup(); 
}

