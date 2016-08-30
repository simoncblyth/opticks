#include "OpticksMgr.hh"

class NConfigurable ; 

#include "Timer.hpp"

#include "Opticks.hh"       // okc-
#include "OpticksEvent.hh"
#include "OpticksHub.hh"    // opticksgeo-
#include "OpticksIdx.hh"    // opticksgeo-

#ifdef WITH_OPTIX
#include "OpViz.hh"     // optixgl-
#include "OpEngine.hh"  // opticksop-
#endif

#define GUI_ 1
#include "OpticksViz.hh"

#include "PLOG.hh"
#include "GGV_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }

OpticksMgr::OpticksMgr(int argc, char** argv) 
    :
    m_opticks(new Opticks(argc, argv)),
    m_hub(new OpticksHub(m_opticks)),
    m_idx(new OpticksIdx(m_hub)),
    m_viz(m_opticks->isCompute() ? NULL : new OpticksViz(m_hub, m_idx)),
#ifdef WITH_OPTIX
    m_ope(new OpEngine(m_hub)),
    m_opv(m_viz ? new OpViz(m_ope,m_viz) : NULL),
#endif
    m_evt(NULL)
{
    m_opticks->Summary("OpticksMgr::OpticksMgr OpticksResource::Summary");
    init();
    initGeometry();
}

void OpticksMgr::init()
{
    m_hub->configure();

    if(m_viz) m_hub->configureState(m_viz->getSceneConfigurable()) ;    // loads/creates Bookmarks
}

void OpticksMgr::initGeometry()
{
    if(m_viz) m_viz->prepareScene();      // setup OpenGL shaders and creates OpenGL context (the window)
 
    m_hub->loadGeometry();                // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry

    if(m_viz) m_viz->uploadGeometry();    // Scene::uploadGeometry, hands geometry to the Renderer instances for upload

#ifdef WITH_OPTIX
    m_ope->prepareOptiX();                // creates OptiX context and populates with geometry by OGeo, OScintillatorLib, ... convert methods 

    if(m_opv) m_opv->prepareTracer();     // creates ORenderer, OTracer
#endif
}

bool OpticksMgr::hasOpt(const char* name)
{
    return m_opticks->hasOpt(name); 
}

NPY<float>* OpticksMgr::loadGenstep()
{
    return m_hub->loadGenstep();
}

void OpticksMgr::createEvent()
{
    m_evt = m_hub->createEvent();
    assert(m_evt == m_hub->getEvent()) ; 
}

void OpticksMgr::propagate(NPY<float>* genstep)
{
    createEvent();
    m_evt->setGenstepData(genstep);

    if(m_viz)
    { 
        m_viz->targetGenstep();       // point Camera at gensteps 
        m_viz->uploadEvent();        // allocates GPU buffers with OpenGL glBufferData
    }

#ifdef WITH_OPTIX
    m_ope->preparePropagator();              // creates OptiX buffers and OBuf wrappers as members of OPropagator

    m_ope->seedPhotonsFromGensteps();        // distributes genstep indices into the photons buffer

    if(hasOpt("dbgseed")) dbgSeed();
    if(hasOpt("onlyseed")) exit(EXIT_SUCCESS);

    m_ope->initRecords();                    // zero records buffer, not working in OptiX 4 in interop 

    if(!hasOpt("nooptix|noevent|nopropagate"))
    {
        m_ope->propagate();                  // perform OptiX GPU propagation 
    }

    indexPropagation();

    if(hasOpt("save"))
    {
        if(m_viz) m_viz->downloadEvent();
        m_ope->saveEvt();
        m_idx->indexEvtOld();
    }
#endif
}

void OpticksMgr::indexPropagation()
{
    if(!m_evt->isIndexed())
    {
#ifdef WITH_OPTIX 
        m_ope->indexSequence();
#endif
        m_idx->indexBoundariesHost();
    }
    if(m_viz) m_viz->indexPresentationPrep();
}

void OpticksMgr::loadPropagation()
{
    LOG(fatal) << "OpticksMgr::loadPropagation" ; 
    createEvent(); 
    m_hub->loadEventBuffers(); // into the above created OpticksEvent

    indexPropagation();

    if(!m_viz) return  ;
    m_viz->targetGenstep();
    m_viz->uploadEvent();  

    LOG(fatal) << "OpticksMgr::loadPropagation DONE" ; 
}

void OpticksMgr::visualize()
{
    if(!m_viz) return ; 
    m_viz->prepareGUI();
    m_viz->renderLoop();    
}

void OpticksMgr::cleanup()
{
#ifdef WITH_OPTIX
    if(m_ope) m_ope->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_opticks->cleanup(); 
}

void OpticksMgr::dbgSeed()
{
#ifdef WITH_OPTIX
    if(!m_ope) return ; 
    OpticksEvent* evt = m_hub->getEvent();    
    NPY<float>* ox = evt->getPhotonData();
    assert(ox);

    // this split between interop and compute mode
    // is annoying, maybe having a shared base protocol
    // for OpEngine and OpticksViz
    // could avoid all the branching 

    if(m_viz) 
    { 
        LOG(info) << "OpticksMgr::debugSeed (interop) download photon seeds " ;
        m_viz->downloadData(ox) ; 
        ox->save("$TMP/dbgseed_interop.npy");
    }
    else
    {
        LOG(info) << "OpticksMgr::debugSeed (compute) download photon seeds " ;
        m_ope->downloadPhotonData();  
        ox->save("$TMP/dbgseed_compute.npy");
    }  
#endif
}



