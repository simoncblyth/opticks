
// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "Composition.hh"
#include "OpticksEvent.hh"

// npy-
#include "Timer.hpp"

// ggeo-
#include "GGeo.hh"


// opticksgeo-
#include "OpticksHub.hh"

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
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }


OpEngine::OpEngine(OpticksHub* hub) 
     : 
      m_timer(NULL),
      m_hub(hub),
      m_opticks(hub->getOpticks()),
      m_fcfg(NULL),
      m_ggeo(hub->getGGeo()),
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


/*
void OpEngine::setEvent(OpticksEvent* evt)
{
    m_evt = evt ;
    m_imp->setEvent(evt); 
}
OpticksEvent* OpEngine::getEvent()
{
    return m_imp->getEvent();
}
*/


void OpEngine::init()
{
    m_imp = new OEngineImp(m_hub);

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

    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();

    OpSeeder* seeder = new OpSeeder(m_hub, ocontext) ; 
    seeder->setPropagator(opropagator);  // only used in compute mode
    seeder->seedPhotonsFromGensteps();
}

void OpEngine::downloadPhotonData()
{
    m_imp->downloadPhotonData();
}



void OpEngine::initRecords()
{
    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();

    OpZeroer* zeroer = new OpZeroer(m_hub, ocontext) ; 
    zeroer->setPropagator(opropagator);  // only used in compute mode

    if(m_opticks->hasOpt("dbginterop"))
    {
        LOG(info) << "OpEngine::initRecords skip OpZeroer::zeroRecords as dbginterop " ; 
    }
    else
    {
        zeroer->zeroRecords();   
        // zeros on GPU record buffer via OptiX or OpenGL
    }
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
    LOG(info) << "OpEngine::indexSequence proceeding  " ;

    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();

    OpIndexer* indexer = new OpIndexer(m_hub, ocontext);
    //indexer->setVerbose(hasOpt("indexdbg"));
    indexer->setPropagator(opropagator);
    indexer->indexSequence();
    indexer->indexBoundaries();

    TIMER("indexSequence"); 
}


void OpEngine::cleanup()
{
     m_imp->cleanup();
}

