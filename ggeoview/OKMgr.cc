#include "OKMgr.hh"

class NConfigurable ; 

#include "SLog.hh"
#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"

#include "OpticksHub.hh"    // okg-
#include "OpticksIdx.hh"    
#include "OpticksGen.hh"    
#include "OpticksRun.hh"    

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
    m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
    m_idx(new OpticksIdx(m_hub)),
    m_gen(m_hub->getGen()),
    m_run(m_hub->getRun()),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
#ifdef WITH_OPTIX
    m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
#endif
    m_count(0)
{
    init();
    (*m_log)("DONE");
}

OKMgr::~OKMgr()
{
    cleanup();
}

void OKMgr::init()
{
    bool g4gun = m_ok->getSourceCode() == G4GUN ;
    if(g4gun)
         LOG(fatal) << "OKMgr doesnt support G4GUN, other that via loading (TO BE IMPLEMENTED) " ;
    assert(!g4gun);
}


void OKMgr::propagate()
{
    int multi = m_ok->getMultiEvent();

    if(m_ok->hasOpt("load"))
    {
         m_run->loadEvent(); 

#ifdef WITH_OPTIX
         m_propagator->uploadEvent();
#endif
         if(m_viz) 
         {
             m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

             m_viz->uploadEvent();      // not needed when propagating as event is created directly on GPU

             m_viz->indexPresentationPrep();
         }
    }
    else if(m_ok->hasOpt("nopropagate"))
    {
        LOG(info) << "--nopropagate/-P" ;
    }
    else if(multi > 0)
    {
#ifdef WITH_OPTIX
        for(int i=0 ; i < multi ; i++) 
        {
            m_run->createEvent();

            m_run->setGensteps(m_gen->getInputGensteps()); 

            m_propagator->propagate();

            if(m_ok->hasOpt("save"))
            {
                 m_run->saveEvent();
            }
        }
#endif
    }
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


