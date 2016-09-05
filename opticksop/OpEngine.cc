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



OpEngine::OpEngine(OpticksHub* hub, bool immediate) 
     : 
      m_hub(hub),
      m_imp(new OEngineImp(m_hub)),
      m_immediate(immediate)
{
   init();
}

void OpEngine::init()
{
    if(m_immediate)
    {
        prepareOptiX();
    }
}

OContext* OpEngine::getOContext()
{
    return m_imp->getOContext(); 
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

    if(m_hub->hasOpt("dbginterop"))
    {
        LOG(info) << "OpEngine::initRecords skip OpZeroer::zeroRecords as dbginterop " ; 
    }
    else
    {
        zeroer->zeroRecords();   // zeros on GPU record buffer via OptiX or OpenGL
    }
}

void OpEngine::propagate()
{
    m_imp->propagate();
}

void OpEngine::downloadEvt()
{
    m_imp->downloadEvt();
}

void OpEngine::indexSequence()
{
   // TODO: reuse this object, for multi-event

    LOG(info) << "OpEngine::indexSequence proceeding  " ;

    OContext* ocontext = m_imp->getOContext();
    OPropagator* opropagator = m_imp->getOPropagator();

    OpIndexer* indexer = new OpIndexer(m_hub, ocontext);
    indexer->setPropagator(opropagator);
    indexer->indexSequence();
    indexer->indexBoundaries();
}

void OpEngine::cleanup()
{
     m_imp->cleanup();
}

