#include <cstring>

template <typename T> class NPY ;
class Scene ; 

#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#include "PLOG.hh"

#ifdef WITH_OPTIX
#include "OpViz.hh"     // optixgl-
#include "OpEngine.hh"  // opticksop-
#endif

// ggeoview-
#include "App.hh"

#define GUI_ 1
#include "OpticksViz.hh"

#include "GGV_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }

App::App(int argc, char** argv )
   : 
      m_opticks(new Opticks(argc, argv)),
      m_hub(new OpticksHub(m_opticks)),
      m_idx(new OpticksIdx(m_hub)),
#ifdef WITH_OPTIX
      m_ope(NULL),
      m_opv(NULL),
#endif
      m_viz(m_opticks->isCompute() ? NULL : new OpticksViz(m_hub, m_idx))
{
     init();
}

bool App::isCompute(){ return m_opticks->isCompute() ; }
bool App::isExit(){    return m_opticks->isExit() ;  }
bool App::hasOpt(const char* name){ return m_hub->hasOpt(name); }

void App::init()
{
    m_opticks->Summary("App::init OpticksResource::Summary");
}

void App::configure()
{
    m_hub->configure(); 
    if(m_viz) m_viz->configure();
    TIMER("configure");
}

void App::prepareViz()
{
    if(!m_viz) return ; 
    m_hub->prepareViz();
    m_viz->prepareScene();
    TIMER("prepareViz");
} 

void App::loadGeometry()
{
    if(hasOpt("nogeometry"))
    { 
        LOG(warning) << "--nogeometry SKIP geometry" ; 
        return ; 
    }
    m_hub->loadGeometry();
}

void App::loadGenstep()
{
    NPY<float>* gs = m_hub->loadGenstep();

    createEvent();
    OpticksEvent* evt = m_hub->getEvent();  

    evt->setGenstepData(gs);  // must do this before can targetGenstep

}

void App::createEvent()
{
    OpticksEvent* evt = m_hub->createEvent();  

    if(m_viz) m_viz->setEvent(evt);
}


void App::loadEvtFromFile()
{
    createEvent();
    m_hub->loadEvent();
}

void App::uploadGeometryViz()
{
    if(!m_viz) return ; 

    if(hasOpt("nogeometry"))
    { 
        LOG(warning) << "--nogeometry SKIP geometry" ; 
        return ; 
    }

    m_viz->uploadGeometry();
    TIMER("uploadGeometryViz"); 
}
void App::targetViz()
{
    if(!m_viz) return ; 
    m_viz->targetGenstep();
    TIMER("targetViz"); 
}
void App::uploadEvtViz()
{
    if(!m_viz) return ; 
    m_viz->uploadEvent();
    TIMER("uploadEvtViz"); 
}
void App::indexPresentationPrep()
{
    if(!m_viz) return ; 
    m_viz->indexPresentationPrep();
    TIMER("indexPresentationPrep"); 
}

void App::prepareGUI()
{
    if(m_viz) m_viz->prepareGUI();
}
void App::renderLoop()
{
    if(!m_viz) return ; 
    m_viz->renderLoop();    
}



void App::indexEvt()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt) return ; 

    if(evt->isIndexed())
    {
        LOG(info) << "App::indexEvt skip as already indexed " ;
        return ; 
    }

#ifdef WITH_OPTIX 
    LOG(info) << "App::indexEvt WITH_OPTIX" ; 
    indexSequence();
    LOG(info) << "App::indexEvt WITH_OPTIX DONE" ; 
#endif

    m_idx->indexBoundariesHost();

    TIMER("indexEvt"); 
}


void App::indexEvtOld()
{
    m_idx->indexEvtOld();
}



void App::cleanup()
{
#ifdef WITH_OPTIX
    if(m_ope) m_ope->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_opticks->cleanup(); 
}


#ifdef WITH_OPTIX
void App::prepareOptiX()
{
    m_ope = new OpEngine(m_opticks, m_hub->getGGeo());   // places geometry into OptiX context with OGeo 
    m_ope->prepareOptiX();

    if(!m_viz) return ; 
    m_opv = new OpViz(m_ope, m_viz->getScene() );         // creates ORenderer, OTracer
    m_viz->setExternalRenderer(m_opv);
}


void App::setupEventInEngine()
{
    if(!m_ope) return ; 
    OpticksEvent* evt = m_hub->getEvent();
    m_ope->setEvent(evt);     // needed for indexing
}

void App::preparePropagator()
{
    if(!m_ope) return ; 
    m_ope->preparePropagator();
}

void App::seedPhotonsFromGensteps()
{
    if(!m_ope) return ; 
    m_ope->seedPhotonsFromGensteps();
    if(hasOpt("dbgseed")) dbgSeed();
}

void App::initRecords()
{
    if(!m_ope) return ; 
    m_ope->initRecords();
}

void App::propagate()
{
    if(!m_ope) return ; 
    if(hasOpt("nooptix|noevent|nopropagate")) return ; 
    m_ope->propagate();
}

void App::saveEvt()
{
    if(!m_ope) return ; 
    if(m_viz) m_viz->downloadEvent();
    m_ope->saveEvt();
}

void App::indexSequence()
{
    if(!m_ope) return ; 
    m_ope->indexSequence();
    LOG(info) << "App::indexSequence DONE" ;
}

void App::dbgSeed()
{
    if(!m_ope) return ; 
    OpticksEvent* evt = m_ope->getEvent();    
    NPY<float>* ox = evt->getPhotonData();
    assert(ox);

    // this split between interop and compute mode
    // is annoying, maybe having a shared base protocol
    // for OpEngine and OpticksViz
    // could avoid all the branching 

    if(m_viz) 
    { 
        LOG(info) << "App::debugSeed (interop) download photon seeds " ;
        m_viz->downloadData(ox) ; 
        ox->save("$TMP/dbgseed_interop.npy");
    }
    else
    {
        LOG(info) << "App::debugSeed (compute) download photon seeds " ;
        m_ope->downloadPhotonData();  
        ox->save("$TMP/dbgseed_compute.npy");
    }  
}

#endif

