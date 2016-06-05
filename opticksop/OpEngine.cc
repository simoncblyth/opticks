
// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"
#include "Composition.hh"
#include "OpticksEvent.hh"

// npy-
#include "NLog.hpp"
#include "Timer.hpp"

// ggeo-
#include "GGeo.hh"

// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
#include "OContext.hh"
#include "OColors.hh"
#include "OGeo.hh"
#include "OBndLib.hh"
#include "OScintillatorLib.hh"
#include "OSourceLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"
#include "OTracer.hh"
#include "OPropagator.hh"


#define TIMER(s) \
    { \
       (*m_timer)((s)); \
       if(m_evt)\
       {\
          Timer& t = *(m_evt->getTimer()) ;\
          t((s)) ;\
       }\
    }


void OpEngine::init()
{
    m_fcfg = m_opticks->getCfg();

    m_timer      = new Timer("OpEngine::");
    m_timer->setVerbose(true);
    m_timer->start();
}


void OpEngine::prepareOptiX()
{
    LOG(info) << "OpEngine::prepareOptiX START" ;  

    std::string builder_   = m_fcfg->getBuilder(); 
    std::string traverser_ = m_fcfg->getTraverser(); 
    const char* builder   = builder_.empty() ? NULL : builder_.c_str() ;
    const char* traverser = traverser_.empty() ? NULL : traverser_.c_str() ;


    OContext::Mode_t mode = m_opticks->isCompute() ? OContext::COMPUTE : OContext::INTEROP ; 

    optix::Context context = optix::Context::create();

    LOG(info) << "OpEngine::prepareOptiX (OContext)" ;
    m_ocontext = new OContext(context, mode); 
    m_ocontext->setStackSize(m_fcfg->getStack());
    m_ocontext->setPrintIndex(m_fcfg->getPrintIndex().c_str());
    m_ocontext->setDebugPhoton(m_fcfg->getDebugIdx());

    LOG(info) << "OpEngine::prepareOptiX (OColors)" ;
    m_ocolors = new OColors(context, m_opticks->getColors() );
    m_ocolors->convert();

    // formerly did OBndLib here, too soon

    LOG(info) << "OpEngine::prepareOptiX (OScintillatorLib)" ;
    m_oscin = new OScintillatorLib(context, m_ggeo->getScintillatorLib());
    m_oscin->convert(); 

    LOG(info) << "OpEngine::prepareOptiX (OSourceLib)" ;
    m_osrc = new OSourceLib(context, m_ggeo->getSourceLib());
    m_osrc->convert(); 

    LOG(info) << "OpEngine::prepareOptiX (OGeo)" ;
    m_ogeo = new OGeo(m_ocontext, m_ggeo, builder, traverser);
    m_ogeo->setTop(m_ocontext->getTop());
    m_ogeo->convert(); 


    LOG(info) << "OpEngine::prepareOptiX (OBndLib)" ;
    m_olib = new OBndLib(context,m_ggeo->getBndLib());
    m_olib->convert(); 
    // this creates the BndLib dynamic buffers, which needs to be after OGeo
    // as that may add boundaries when using analytic geometry


    LOG(debug) << m_ogeo->description("OpEngine::prepareOptiX ogeo");
    LOG(info) << "OpEngine::prepareOptiX DONE" ;  

}

void OpEngine::preparePropagator()
{
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    int  override   = m_fcfg->getOverride();

    if(!m_evt) return ; 

    assert(!noevent);

    m_opropagator = new OPropagator(m_ocontext, m_opticks);

    m_opropagator->setEvent(m_evt);

    m_opropagator->setTrivial(trivial);
    m_opropagator->setOverride(override);

    m_opropagator->initRng();
    m_opropagator->initEvent();

    LOG(info) << "OpEngine::preparePropagator DONE "; 
}



void OpEngine::seedPhotonsFromGensteps()
{
    if(!m_evt) return ; 

    OpSeeder* seeder = new OpSeeder(m_ocontext) ; 

    seeder->setEvent(m_evt);
    seeder->setPropagator(m_opropagator);  // only used in compute mode

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

    OpZeroer* zeroer = new OpZeroer(m_ocontext) ; 

    zeroer->setEvent(m_evt);
    zeroer->setPropagator(m_opropagator);  // only used in compute mode

    zeroer->zeroRecords();   
    // zeros on GPU record buffer via OptiX or OpenGL
}


void OpEngine::propagate()
{
    LOG(info)<< "OpEngine::propagate" ;

    m_opropagator->prelaunch();     
    TIMER("prelaunch"); 

    m_opropagator->launch();     
    TIMER("propagate"); 

    m_opropagator->dumpTimes("OpEngine::propagate");
}



void OpEngine::saveEvt()
{
    if(!m_evt) return ; 

    if(m_opticks->isCompute())
    {
        m_opropagator->downloadEvent();
    }
    else
    {
        //Rdr::download(m_evt);   now done from App::saveEvt
    }

    TIMER("downloadEvt"); 

    m_evt->dumpDomains("OpEngine::saveEvt dumpDomains");
    m_evt->save(true);
 
    TIMER("saveEvt"); 
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

    OpIndexer* indexer = new OpIndexer(m_ocontext);
    //indexer->setVerbose(hasOpt("indexdbg"));
    indexer->setEvent(m_evt);
    indexer->setPropagator(m_opropagator);

    indexer->indexSequence();
    indexer->indexBoundaries();

    TIMER("indexSequence"); 
}


void OpEngine::cleanup()
{
    if(m_ocontext) m_ocontext->cleanUp();
}


