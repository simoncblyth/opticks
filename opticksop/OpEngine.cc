
// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "Composition.hh"
#include "OpticksEvent.hh"

// npy-
#include "Timer.hpp"

// ggeo-
#include "GGeo.hh"

// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
#include "OEngineImp.hh"

#include "PLOG.hh"


#define TIMER(s) \
    { \
       (*m_timer)((s)); \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpEngine::OpEngine(Opticks* opticks, GGeo* ggeo) 
     : 
      m_timer(NULL),
      m_opticks(opticks),
      m_fcfg(NULL),
      m_ggeo(ggeo),
      m_evt(NULL),
      m_imp(NULL)
{
      init();
}


Opticks* OpEngine::getOpticks()
{
    return m_opticks ; 
}
OContext* OpEngine::getOContext()
{
    return m_imp->getOContext(); 
}

void OpEngine::setEvent(OpticksEvent* evt)
{
    m_evt = evt ;
    m_imp->setEvent(evt); 
}


void OpEngine::init()
{
    m_imp = new OEngineImp(m_opticks, m_ggeo);

    m_fcfg = m_opticks->getCfg();

    m_timer      = new Timer("OpEngine::");
    m_timer->setVerbose(true);
    m_timer->start();
}


void OpEngine::prepareOptiX()
{
    LOG(info) << "OpEngine::prepareOptiX START" ;  
    m_imp->prepareOptiX();
    LOG(info) << "OpEngine::prepareOptiX DONE" ;  
}

void OpEngine::preparePropagator()
{
    LOG(info) << "OpEngine::preparePropagator START "; 
    m_imp->preparePropagator();
    LOG(info) << "OpEngine::preparePropagator DONE "; 
}



void OpEngine::seedPhotonsFromGensteps()
{
    if(!m_evt) return ; 


    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();


    OpSeeder* seeder = new OpSeeder(ocontext) ; 

    seeder->setEvent(m_evt);
    seeder->setPropagator(opropagator);  // only used in compute mode

    seeder->seedPhotonsFromGensteps();
}


void OpEngine::initRecords()
{
    if(!m_evt) return ; 

    if(!m_evt->isStep())
    {
        LOG(info) << "OpEngine::initRecords --nostep mode skipping " ;
        return ; 
    }


    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();

    OpZeroer* zeroer = new OpZeroer(ocontext) ; 

    zeroer->setEvent(m_evt);
    zeroer->setPropagator(opropagator);  // only used in compute mode

    zeroer->zeroRecords();   
    // zeros on GPU record buffer via OptiX or OpenGL
}


void OpEngine::propagate()
{
    m_imp->propagate();
}

void OpEngine::saveEvt()
{
    m_imp->saveEvt();
}



void OpEngine::indexSequence()
{
    if(!m_evt)
    { 
       LOG(warning) << "OpEngine::indexSequence NULL evt : skipping  " ;
       return ; 
    }
    if(!m_evt->isStep())
    {
        LOG(info) << "OpEngine::indexSequence --nostep mode skipping " ;
        return ; 
    }

    LOG(info) << "OpEngine::indexSequence proceeding  " ;

    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();

    OpIndexer* indexer = new OpIndexer(ocontext);
    //indexer->setVerbose(hasOpt("indexdbg"));
    indexer->setEvent(m_evt);
    indexer->setPropagator(opropagator);

    indexer->indexSequence();
    indexer->indexBoundaries();

    TIMER("indexSequence"); 
}


void OpEngine::cleanup()
{
     m_imp->cleanup();
}

