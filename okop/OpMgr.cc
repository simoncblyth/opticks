#include "OpMgr.hh"

class NConfigurable ; 


#include "PLOG.hh"


#include "SLog.hh"
#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"

#include "OpticksHub.hh"    // okg-
#include "OpticksIdx.hh"    
#include "OpticksGen.hh"    
#include "OpticksRun.hh"    

#include "OpEvt.hh"    

#include "OpPropagator.hh"  // okop-


#define TIMER(s) \
    { \
       if(m_ok)\
       {\
          Timer& t = *(m_ok->getTimer()) ;\
          t((s)) ;\
       }\
    }



// even though may end up doing the geocache check inside OpticksHub tis 
// convenient to have the Opticks instance outside OpMgr 

OpMgr::OpMgr(Opticks* ok ) 
    :
    m_log(new SLog("OpMgr::OpMgr","",fatal)),
    m_ok(ok ? ok : Opticks::GetInstance()),         
    m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry OR adopt a preexisting GGeo instance
    m_idx(new OpticksIdx(m_hub)),
    m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
    m_gen(m_hub->getGen()),
    m_run(m_hub->getRun()),
    m_propagator(new OpPropagator(m_hub, m_idx)),
    m_count(0),
    m_opevt(NULL)
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


void OpMgr::setGensteps(NPY<float>* gensteps)
{
    m_gensteps = gensteps ; 
}

OpticksEvent* OpMgr::getEvent() const 
{
    return m_run->getEvent() ; 
}
OpticksEvent* OpMgr::getG4Event() const 
{
    return m_run->getG4Event() ; 
}





void OpMgr::propagate()
{
    const Opticks& ok = *m_ok ; 
    
    if(ok("nopropagate")) return ; 

    assert( ok.isEmbedded() ); 

    assert( m_gensteps ); 

    bool production = m_ok->isProduction();

    bool compute = true ; 

    m_gensteps->setBufferSpec(OpticksEvent::GenstepSpec(compute));

    m_run->createEvent(0);

    m_run->setGensteps(m_gensteps); 

    m_propagator->propagate();

    if(ok("save")) 
    {
        m_run->saveEvent();
        if(!production) m_hub->anaEvent();
    }

    m_ok->postpropagate();  // profiling 
}


void OpMgr::reset()
{   
    m_run->resetEvent();

    // m_opevt->resetGensteps();  ???
}





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




