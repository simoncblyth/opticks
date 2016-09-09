#include "OKG4Mgr.hh"

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
    m_log(new SLog("OKG4Mgr::OKG4Mgr")),
    m_ok(new Opticks(argc, argv, true)),               // true: integrated running 
    m_hub(new OpticksHub(m_ok)),                       // configure, loadGeometry and setupInputGensteps immediately
    m_idx(new OpticksIdx(m_hub)),
    m_gen(m_hub->getGen()),
    m_run(m_hub->getRun()),
    m_g4(new CG4(m_hub, true)),                        // true: configure and initialize immediately 
    m_collector(new CCollector(m_hub)),                // after CG4 loads geometry, currently hub just used for material code lookup, not evt access
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
#ifdef WITH_OPTIX
    m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
#endif
    m_placeholder(0)
{
    init();
    (*m_log)("DONE");
}

OKG4Mgr::~OKG4Mgr()
{
    cleanup();
}


void OKG4Mgr::init()
{
}


void OKG4Mgr::propagate()
{
    int multi = m_ok->getMultiEvent();

    if(m_ok->hasOpt("load"))
    {   
         m_run->loadEvent(); 

         if(m_ok->isExit()) exit(EXIT_FAILURE) ; 

         if(m_viz) 
         {   
             m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

             m_viz->uploadEvent();      // not needed when propagating as event is created directly on GPU
         }   
    }   
    else if(m_ok->hasOpt("nopropagate"))
    {   
        LOG(info) << "--nopropagate/-P" ;
    }   
    else if( m_ok->isLiveGensteps() )   // eg G4GUN running 
    {
        m_run->createEvent();

        m_g4->propagate();
   
        NPY<float>* gs = m_collector->getGensteps() ;   // TODO: come from g4evt not collector

        m_run->setGensteps(gs); 

        m_propagator->propagate();

        if(m_ok->hasOpt("save"))
        {   
            m_run->saveEvent();
        }   

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

