
#include "SLog.hh"

#include "Opticks.hh"  // okc-
#include "OpticksHub.hh" // okg-

// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
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
      m_seeder(new OpSeeder(m_hub, m_imp)),
      m_zeroer(new OpZeroer(m_hub, m_imp)),
      m_indexer(new OpIndexer(m_hub, m_imp))
{
   init();
}

void OpEngine::init()
{
   
}


void OpEngine::propagate()
{
    m_imp->initEvent();                   // creates OptiX buffers, uploads gensteps

    m_seeder->seedPhotonsFromGensteps();  // distributes genstep indices into the photons buffer

    m_zeroer->zeroRecords();              // zeros on GPU record buffer via OptiX or OpenGL  (not working OptiX 4 in interop)

    m_imp->propagate();                   // perform OptiX GPU propagation 

    m_indexer->indexSequence();

    m_indexer->indexBoundaries();
}



void OpEngine::downloadEvt()
{
    m_imp->downloadEvt();
}
void OpEngine::downloadPhotonData()  // was used for debugging of seeding (buffer overwrite in interop mode on Linux)
{
    m_imp->downloadPhotonData();
}
void OpEngine::cleanup()
{
    m_imp->cleanup();
}



void OpEngine::Summary(const char* msg)
{
    LOG(info) << msg ; 
}


