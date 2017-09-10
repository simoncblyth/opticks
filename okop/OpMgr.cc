#include "OpMgr.hh"

class NConfigurable ; 

#include "SLog.hh"
#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"

#include "OpticksHub.hh"    // okg-
#include "OpticksIdx.hh"    
#include "OpticksGen.hh"    
#include "OpticksRun.hh"    


#include "OpPropagator.hh"  // okop-

#include "PLOG.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpMgr::OpMgr(int argc, char** argv, const char* argforced ) 
    :
    m_log(new SLog("OpMgr::OpMgr")),
    m_ok(new Opticks(argc, argv, argforced)),         
    m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
    m_idx(new OpticksIdx(m_hub)),
    m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
    m_gen(m_hub->getGen()),
    m_run(m_hub->getRun()),
    m_propagator(new OpPropagator(m_hub, m_idx)),
    m_count(0)
{
    init();
    (*m_log)("DONE");
}

OpMgr::~OpMgr()
{
    cleanup();
}

void OpMgr::init()
{
    bool g4gun = m_ok->getSourceCode() == G4GUN ;
    if(g4gun)
         LOG(fatal) << "OpMgr doesnt support G4GUN, other that via loading (TO BE IMPLEMENTED) " ;
    assert(!g4gun);

    m_ok->dumpParameters("OpMgr::init");
}


/*
void OpMgr::propagate()
{
    const Opticks& ok = *m_ok ; 
    
    if(ok("nopropagate")) return ; 

    bool production = m_ok->isProduction();

    if(ok.isLoad())
    {
         m_run->loadEvent(); 

         m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

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
*/


void OpMgr::cleanup()
{
   // m_propagator->cleanup();
    m_hub->cleanup();
    m_ok->cleanup(); 
}


void OpMgr::snap()
{
    LOG(info) << "OpMgr::snap" ; 
    m_propagator->snap(); 
}


