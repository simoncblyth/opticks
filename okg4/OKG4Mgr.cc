#include "OKG4Mgr.hh"

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

#include "CG4.hh"


#include "PLOG.hh"
#include "OKG4_BODY.hh"

#define TIMER(s) \
    { \
       if(m_hub)\
       {\
          Timer& t = *(m_hub->getTimer()) ;\
          t((s)) ;\
       }\
    }

OKG4Mgr::OKG4Mgr(int argc, char** argv) 
    :
    m_ok(new Opticks(argc, argv)),
    m_hub(new OpticksHub(m_ok)),
    m_idx(new OpticksIdx(m_hub)),
    m_g4(new CG4(m_hub)),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx)),
#ifdef WITH_OPTIX
    m_ope(new OpEngine(m_hub)),
    m_opv(m_viz ? new OpViz(m_ope,m_viz) : NULL),
#endif
    m_placeholder(0)
{
    m_ok->setIntegrated(true);  // when integrated itag (default 100) is used by save/load to distinguish against priors 
    m_ok->Summary("OKG4Mgr::OKG4Mgr OpticksResource::Summary");
    init();
    initGeometry();
}

void OKG4Mgr::init()
{
    m_hub->configure();

    if(m_viz) m_hub->configureState(m_viz->getSceneConfigurable()) ;    // loads/creates Bookmarks

    m_g4->configure();

}

void OKG4Mgr::initGeometry()
{
    m_g4->initialize();

    m_hub->setMaterialMap(m_g4->getMaterialMap());
    
    if(m_viz) m_viz->prepareScene();      // setup OpenGL shaders and creates OpenGL context (the window)
 
    m_hub->loadGeometry();                // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry

    if(m_viz) m_viz->uploadGeometry();    // Scene::uploadGeometry, hands geometry to the Renderer instances for upload

#ifdef WITH_OPTIX
    m_ope->prepareOptiX();                // creates OptiX context and populates with geometry by OGeo, OScintillatorLib, ... convert methods 

    if(m_opv) m_opv->prepareTracer();     // creates ORenderer, OTracer
#endif
}


void OKG4Mgr::propagate()
{
    m_g4->propagate();

    NPY<float>* gs = m_g4->getGensteps();

    unsigned ngs = gs->getNumItems();
    if(ngs == 0)
    {
        LOG(warning) << "OKG4Mgr::propagate"
                     << " SKIPPING as there are zero optical gensteps (ie Cerenkov or scintillation gensteps) "
                     << " use --g4gun option to produce some "
                     ;
        return ;  
    }


    m_hub->translateGensteps(gs);     // relabel and apply lookup

    OpticksEvent* evt = m_hub->createEvent();
    evt->setGenstepData(gs);
    LOG(info) << "OpticksEvent tagdir : " << evt->getTagDir() ;  

    if(m_viz)
    { 
        m_viz->targetGenstep();       // point Camera at gensteps 
        m_viz->uploadEvent();        // allocates GPU buffers with OpenGL glBufferData
    }

#ifdef WITH_OPTIX
    m_ope->preparePropagator();              // creates OptiX buffers and OBuf wrappers as members of OPropagator

    m_ope->seedPhotonsFromGensteps();        // distributes genstep indices into the photons buffer

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
        m_ope->saveEvt();
        m_idx->indexEvtOld();
    }
#endif
}




void OKG4Mgr::indexPropagation()
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

void OKG4Mgr::visualize()
{
    if(!m_viz) return ; 
    m_viz->prepareGUI();
    m_viz->renderLoop();    
}

void OKG4Mgr::cleanup()
{
#ifdef WITH_OPTIX
    if(m_ope) m_ope->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
    m_g4->cleanup(); 
}

