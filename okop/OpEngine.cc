
#include "SLog.hh"

#include "Opticks.hh"  // okc-
#include "OpticksHub.hh" // okg-
#include "OpticksSwitches.h" 

// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
#include "OConfig.hh"
#include "OContext.hh"
#include "OEvent.hh"
#include "OPropagator.hh"
#include "OScene.hh"

#include "PLOG.hh"


unsigned OpEngine::getOptiXVersion()
{
   return OConfig::OptiXVersion();
}

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
      m_ocontext(m_scene->getOContext()),
      m_entry(m_ocontext->addEntry(m_ok->getEntryCode())),
      m_oevt(new OEvent(m_hub, m_ocontext)),
      m_propagator(new OPropagator(m_hub, m_oevt, m_entry)),
      m_seeder(new OpSeeder(m_hub, m_oevt)),
      m_zeroer(new OpZeroer(m_hub, m_oevt)),
      m_indexer(new OpIndexer(m_hub, m_oevt))
{
   m_ok->setOptiXVersion(OConfig::OptiXVersion()); 
   (*m_log)("DONE");
}


// NB OpEngine is ONLY AT COMPUTE LEVEL, FOR THE FULL PICTURE NEED TO SEE ONE LEVEL UP 
//     OKPropagator::uploadEvent 
//     OKPropagator::downloadEvent
//   

void OpEngine::uploadEvent()
{
    m_oevt->upload();                   // creates OptiX buffers, uploads gensteps
}

void OpEngine::propagate()
{
    m_seeder->seedPhotonsFromGensteps();  // distributes genstep indices into the photons buffer OR seed buffer

#ifdef WITH_SEED_BUFFER
#else
    m_oevt->markDirtyPhotonBuffer();      // inform OptiX that must sync with the CUDA modified photon buffer
#endif

    //m_zeroer->zeroRecords();              // zeros on GPU record buffer via OptiX or OpenGL  (not working OptiX 4 in interop)

    m_propagator->launch();               // perform OptiX GPU propagation : write the photon, record and sequence buffers

#ifdef WITH_RECORD
    m_indexer->indexSequence();
#endif

    m_indexer->indexBoundaries();
}

void OpEngine::downloadEvent()
{
    m_oevt->download();
}

void OpEngine::cleanup()
{
    m_scene->cleanup();
}

void OpEngine::Summary(const char* msg)
{
    LOG(info) << msg ; 
}



void OpEngine::downloadPhotonData()  // was used for debugging of seeding (buffer overwrite in interop mode on Linux)
{
     if(m_ok->isCompute()) m_oevt->downloadPhotonData(); 
}

