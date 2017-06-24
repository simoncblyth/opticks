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

#include "OKPropagator.hh"  // ggeoview-

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


OKMgr::OKMgr(int argc, char** argv, const char* argforced ) 
    :
    m_log(new SLog("OKMgr::OKMgr")),
    m_ok(new Opticks(argc, argv, argforced)),         
    m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
    m_idx(new OpticksIdx(m_hub)),
    m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
    m_gen(m_hub->getGen()),
    m_run(m_hub->getRun()),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
    m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
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

    m_ok->dumpParameters("OKMgr::init");
}


void OKMgr::propagate()
{
    const Opticks& ok = *m_ok ; 
    
    if(ok("nopropagate")) return ; 

    bool production = m_ok->isProduction();

    if(ok.isLoad())
    {
         m_run->loadEvent(); 

         if(m_viz) 
         {
             m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

             m_viz->uploadEvent();

             m_viz->indexPresentationPrep();
         }
    }
    else if(m_num_event > 0)
    {
        for(int i=0 ; i < m_num_event ; i++) 
        {
            m_run->createEvent(i);

            m_run->setGensteps(m_gen->getInputGensteps()); 

            m_propagator->propagate();

            if(ok("save")) 
            {
                m_run->saveEvent();
                if(!production) m_hub->anaEvent();
            }

            m_run->resetEvent();
        }

        m_ok->postpropagate();
    }
}

void OKMgr::visualize()
{
    if(m_viz) m_viz->visualize();
}

void OKMgr::cleanup()
{
    m_propagator->cleanup();
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
}


