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
    m_ok(new Opticks(argc, argv)),
    m_hub(new OpticksHub(m_ok)),
    m_idx(new OpticksIdx(m_hub)),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx)),
#ifdef WITH_OPTIX
    m_ope(new OpEngine(m_hub)),
    m_opv(m_viz ? new OpViz(m_ope,m_viz) : NULL),
#endif
    m_placeholder(0)
{
    m_ok->Summary("OpticksMgr::OpticksMgr OpticksResource::Summary");
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
   return m_ok->hasOpt(name);
}


NPY<float>* OpticksMgr::loadGenstep()
{
    return m_hub->loadGenstep();
}

void OpticksMgr::propagate(NPY<float>* genstep)
{
    m_hub->initOKEvent(genstep);

    if(m_viz)
    { 
        m_viz->targetGenstep();       // point Camera at gensteps 
        m_viz->uploadEvent();        // allocates GPU buffers with OpenGL glBufferData
    }

#ifdef WITH_OPTIX
    m_ope->preparePropagator();              // creates OptiX buffers and OBuf wrappers as members of OPropagator

    m_ope->seedPhotonsFromGensteps();        // distributes genstep indices into the photons buffer

    if(m_ok->hasOpt("dbgseed")) dbgSeed();
    if(m_ok->hasOpt("onlyseed")) exit(EXIT_SUCCESS);

    m_ope->initRecords();                    // zero records buffer, not working in OptiX 4 in interop 

    if(!m_ok->hasOpt("nooptix|noevent|nopropagate"))
    {
        m_ope->propagate();                  // perform OptiX GPU propagation 
    }

    indexPropagation();

    if(m_ok->hasOpt("save"))
    {
        if(m_viz) m_viz->downloadEvent();
        m_ope->downloadEvt();
        m_idx->indexEvtOld();

        OpticksEvent* okevt = m_hub->getOKEvent();
        okevt->dumpDomains("OpticksMgr::propagate okevt domains");
        okevt->save();
    }
#endif
}

void OpticksMgr::indexPropagation()
{
    OpticksEvent* evt = m_hub->getEvent();
    if(!evt->isIndexed())
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
    m_hub->loadPersistedEvent(); 

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
    m_ok->cleanup(); 
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



