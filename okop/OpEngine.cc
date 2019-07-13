
#include "SLog.hh"

#include "Opticks.hh"  // okc-
#include "OpticksEntry.hh" 
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


const plog::Severity OpEngine::LEVEL = PLOG::EnvLevel("OpEngine", "DEBUG") ; 


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


int OpEngine::preinit() const
{
    OKI_PROFILE("_OpEngine::OpEngine");
    return 0 ; 
}

OpEngine::OpEngine(OpticksHub* hub) 
    : 
    m_preinit(preinit()),
    m_log(new SLog("OpEngine::OpEngine","",LEVEL)),
    m_hub(hub),
    m_ok(m_hub->getOpticks()),
    m_scene(new OScene(m_hub)),
    m_ocontext(m_scene->getOContext()),
    m_entry(NULL),
    m_oevt(NULL),
    m_propagator(NULL),
    m_seeder(NULL),
    m_zeroer(NULL),
    m_indexer(NULL)
{
   init();
   (*m_log)("DONE");
}

void OpEngine::init()
{
    m_ok->setOptiXVersion(OConfig::OptiXVersion()); 

    bool is_load = m_ok->isLoad() ; 
    bool is_tracer = m_ok->isTracer() ;

    LOG(LEVEL) 
        << " is_load " << is_load 
        << " is_tracer " << is_tracer
        << " OptiXVersion " << m_ok->getOptiXVersion()
        ; 

    if(is_load)
    {
        LOG(LEVEL) << "skip initPropagation as just loading pre-cooked event " ;
    }
    else if(is_tracer)
    {
        LOG(LEVEL) << "skip initPropagation as tracer mode is active  " ; 
    }
    else
    {
        pLOG(LEVEL,0) << "(" ;  // -1 for one notch more logging 
        initPropagation(); 
        pLOG(LEVEL,0) << ")" ;
    }
    OKI_PROFILE("OpEngine::OpEngine");
}

void OpEngine::initPropagation()
{
    m_entry = m_ocontext->addEntry(m_ok->getEntryCode()) ;
    LOG(LEVEL) << " entry " << m_entry->desc() ; 

    m_oevt = new OEvent(m_ok, m_ocontext);
    m_propagator = new OPropagator(m_ok, m_oevt, m_entry);
    m_seeder = new OpSeeder(m_ok, m_oevt) ;
    m_zeroer = new OpZeroer(m_ok, m_oevt) ;
    m_indexer = new OpIndexer(m_ok, m_oevt) ;
}



 

unsigned OpEngine::uploadEvent()
{
    LOG(info) << "." ; 
    LOG(verbose) << "[" ; 
    unsigned n = m_oevt->upload();                   // creates OptiX buffers, uploads gensteps
    LOG(verbose) << "]" ; 
    return n ; 
}

void OpEngine::propagate()
{
    LOG(info) << "[" ; 

    LOG(debug) << "( seeder.seedPhotonsFromGensteps ";  
    m_seeder->seedPhotonsFromGensteps();  // distributes genstep indices into the photons buffer OR seed buffer
    LOG(debug) << ") seeder.seedPhotonsFromGensteps ";  

    m_oevt->markDirty();                   // inform OptiX that must sync with the CUDA modified photon/seed depending on WITH_SEED_BUFFER 

    //m_zeroer->zeroRecords();              // zeros on GPU record buffer via OptiX or OpenGL  (not working OptiX 4 in interop)

    LOG(info) << "( propagator.launch ";  
    m_propagator->launch();               // perform OptiX GPU propagation : write the photon, record and sequence buffers
    LOG(info) << ") propagator.launch ";  

    indexEvent();
    LOG(info) << "]" ; 
}


/**
OpEngine::indexEvent
---------------------

In production event indexing is skipped.


**/

void OpEngine::indexEvent()
{
    if(m_ok->isProduction()) return ; 

#ifdef WITH_RECORD
    m_indexer->indexSequence();
#endif
    m_indexer->indexBoundaries();
}


unsigned OpEngine::downloadEvent()
{
    LOG(info) << "." ; 
    LOG(debug) << "[" ; 
    unsigned n = m_oevt->download();
    LOG(debug) << "]" ; 
    return n ; 
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

