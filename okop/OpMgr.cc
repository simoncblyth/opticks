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
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }



// even though may end up doing the geocache check inside OpticksHub tis 
// convenient to have the Opticks instance outside OpMgr 

OpMgr::OpMgr(Opticks* ok ) 
    :
    m_log(new SLog("OpMgr::OpMgr")),
    m_ok(ok ? ok : Opticks::GetInstance()),         
    m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
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

void OpMgr::addGenstep( float* data, unsigned num_float )
{
    assert(num_float == 6*4) ;
    if(!m_opevt) m_opevt = new OpEvt ; 
    m_opevt->addGenstep(data, num_float ); 
}

unsigned OpMgr::getNumGensteps() const 
{
   return m_opevt ? m_opevt->getNumGensteps() : 0 ;  
}



unsigned OpMgr::getNumHits() const 
{
    return 0u ; 
}

void OpMgr::propagate()
{
    const Opticks& ok = *m_ok ; 
    
    if(ok("nopropagate")) return ; 

    bool production = m_ok->isProduction();

    if(ok.isLoad())
    {
         m_run->loadEvent(); 

         m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt
        // note that targetting is needed for pure compute not just viz, as snapshots are pure compute 
    }
    else if(ok.isEmbedded())
    {
        NPY<float>* embedded_gensteps = m_opevt ? m_opevt->getEmbeddedGensteps() : NULL ; 

        if(embedded_gensteps)
        {
            embedded_gensteps->setLookup(m_hub->getLookup());

            bool compute = true ; 

            embedded_gensteps->setBufferSpec(OpticksEvent::GenstepSpec(compute));

            m_run->createEvent(0);

            m_run->setGensteps(embedded_gensteps); 

            m_propagator->propagate();

            if(ok("save")) 
            {
                m_run->saveEvent();
                if(!production) m_hub->anaEvent();
            }

            m_run->resetEvent();

            m_opevt->resetGensteps(); 

            m_ok->postpropagate();
        }
        else
        {
            std::cerr << "OpMgr::propagate"
                      << " called with no embedded gensteps collected " 
                      ;
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



void OpMgr::saveEmbeddedGensteps(const char* path) const 
{
    assert(m_opevt);
    m_opevt->saveEmbeddedGensteps(path);
}
void OpMgr::loadEmbeddedGensteps(const char* path)
{
    assert(m_opevt);
    m_opevt->loadEmbeddedGensteps(path);
}

void OpMgr::setLookup(const char* json)
{
    m_hub->overrideMaterialMapA(json);
}


