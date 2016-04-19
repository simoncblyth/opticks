
// opticks-
#include "Opticks.hh"
#include "OpticksCfg.hh"

// npy-
#include "NLog.hpp"
#include "Timer.hpp"
#include "NumpyEvt.hpp"

// ggeo-
#include "GGeo.hh"
#include "GCache.hh"


// opop-
#include "OpEngine.hh"
#include "OpIndexer.hh"
#include "OpSeeder.hh"
#include "OpZeroer.hh"

// optixrap-
#include "OContext.hh"
#include "OColors.hh"
#include "OFrame.hh"
#include "ORenderer.hh"
#include "OGeo.hh"
#include "OBndLib.hh"
#include "OScintillatorLib.hh"
#include "OSourceLib.hh"
#include "OBuf.hh"
#include "OConfig.hh"
#include "OTracer.hh"
#include "OPropagator.hh"

// oglrap-
// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
//   TODO: partition to avoid OpenGL dependency from raw OptiX compute
#include "Frame.hh"
#include "Scene.hh"
#include "Composition.hh"
#include "Interactor.hh"
#include "Renderer.hh"
#include "Rdr.hh"



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
    m_cache = m_ggeo->getCache(); 

    m_timer      = new Timer("OpEngine::");
    m_timer->setVerbose(true);
    m_timer->start();
}


void OpEngine::prepareOptiX()
{
    // TODO: move inside OGeo or new opop-/OpEngine ? 

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
    m_ocolors = new OColors(context, m_cache->getColors() );
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


void OpEngine::postSetScene()
{
    m_composition = m_scene->getComposition();
    m_interactor = m_scene->getInteractor();
}


void OpEngine::prepareOptiXViz()
{
    if(m_opticks->isCompute()) return ; 

    if(!m_scene) return ; 

    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();

    optix::Context context = m_ocontext->getContext();

    m_oframe = new OFrame(context, width, height);

    context["output_buffer"]->set( m_oframe->getOutputBuffer() );

    m_interactor->setTouchable(m_oframe);

    Renderer* rtr = m_scene->getRaytraceRenderer();

    m_orenderer = new ORenderer(rtr, m_oframe, m_scene->getShaderDir(), m_scene->getShaderInclPath());

    m_otracer = new OTracer(m_ocontext, m_composition);

    LOG(info) << "OpEngine::prepareOptiXViz DONE "; 

    m_ocontext->dump("OpEngine::prepareOptiXVix");
}


void OpEngine::render()
{
    if(m_otracer && m_orenderer)
    {
        if(m_composition->hasChangedGeometry())
        {
            unsigned int scale = m_interactor->getOptiXResolutionScale() ; 
            m_otracer->setResolutionScale(scale) ;
            m_otracer->trace();
            m_oframe->push_PBO_to_Texture();           
        }
        else
        {
            // dont bother tracing when no change in geometry
        }
    }
}



void OpEngine::preparePropagator()
{
    bool noevent    = m_fcfg->hasOpt("noevent");
    bool trivial    = m_fcfg->hasOpt("trivial");
    int  override   = m_fcfg->getOverride();

    if(!m_evt) return ; 

    assert(!noevent);

    m_opropagator = new OPropagator(m_ocontext, m_opticks);

    m_opropagator->setNumpyEvt(m_evt);

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

    seeder->setEvt(m_evt);
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

    zeroer->setEvt(m_evt);
    zeroer->setPropagator(m_opropagator);  // only used in compute mode

    zeroer->zeroRecords();
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
        Rdr::download(m_evt);
    }

    TIMER("downloadEvt"); 

    m_evt->dumpDomains("OpEngine::saveEvt dumpDomains");
    m_evt->save(true);
 
    TIMER("saveEvt"); 
}



void OpEngine::indexSequence()
{
    if(!m_evt) return ; 
    if(!m_evt->isStep())
    {
        LOG(info) << "OpEngine::indexSequence --nostep mode skipping " ;
        return ; 
    }

    OpIndexer* indexer = new OpIndexer(m_ocontext);
    //indexer->setVerbose(hasOpt("indexdbg"));
    indexer->setEvt(m_evt);
    indexer->setPropagator(m_opropagator);

    indexer->indexSequence();
    indexer->indexBoundaries();

    TIMER("indexSequence"); 
}


void OpEngine::cleanup()
{
    if(m_ocontext) m_ocontext->cleanUp();
}
