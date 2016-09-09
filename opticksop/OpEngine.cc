
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
#include "OScene.hh"

#include "PLOG.hh"


OContext* OpEngine::getOContext()
{
    return m_scene->getOContext(); 
}

OPropagator* OpEngine::getOPropagator()
{
    return m_propagator ; 
}


OpEngine::OpEngine(OpticksHub* hub) 
     : 
      m_log(new SLog("OpEngine::OpEngine")),
      m_hub(hub),
      m_ok(m_hub->getOpticks()),
      m_scene(new OScene(m_hub)),
      m_propagator(OPropagator::make(m_scene->getOContext(), m_hub)),
      m_seeder(new OpSeeder(m_hub, this)),
      m_zeroer(new OpZeroer(m_hub, this)),
      m_indexer(new OpIndexer(m_hub, this))
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
    m_scene->cleanup();
}



void OpEngine::Summary(const char* msg)
{
    LOG(info) << msg ; 
}


