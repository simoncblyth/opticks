
#include "SLog.hh"

#include "Opticks.hh"  // okc-
#include "OpticksHub.hh" // okg-

// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
#include "OPropagator.hh"
#include "OEngineImp.hh"

#include "PLOG.hh"

OContext* OpEngine::getOContext()
{
    return m_imp->getOContext(); 
}

OpEngine::OpEngine(OpticksHub* hub) 
     : 
      m_log(new SLog("OpEngine::OpEngine")),
      m_hub(hub),
      m_ok(m_hub->getOpticks()),
      m_imp(new OEngineImp(m_hub)),
      m_propagator(m_imp->getOPropagator()),
      m_seeder(new OpSeeder(m_hub, m_imp)),
      m_zeroer(new OpZeroer(m_hub, m_imp)),
      m_indexer(new OpIndexer(m_hub, m_imp))
{
   (*m_log)("DONE");
}


void OpEngine::propagate()
{
    m_seeder->seedPhotonsFromGensteps();  // distributes genstep indices into the photons buffer

    m_zeroer->zeroRecords();              // zeros on GPU record buffer via OptiX or OpenGL  (not working OptiX 4 in interop)

    m_propagator->launch();                   // perform OptiX GPU propagation 

    m_indexer->indexSequence();

    m_indexer->indexBoundaries();
}


void OpEngine::uploadEvent()
{
    m_propagator->uploadEvent();                   // creates OptiX buffers, uploads gensteps
}
void OpEngine::downloadEvent()
{
    m_propagator->downloadEvent();
}

void OpEngine::downloadPhotonData()  // was used for debugging of seeding (buffer overwrite in interop mode on Linux)
{
     if(m_ok->isCompute()) m_propagator->downloadPhotonData(); 
}
void OpEngine::cleanup()
{
    m_imp->cleanup();
}



void OpEngine::Summary(const char* msg)
{
    LOG(info) << msg ; 
}


