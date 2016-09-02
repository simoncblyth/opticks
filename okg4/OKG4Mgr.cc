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
    m_ok(new Opticks(argc, argv, true)),  // true: integrated running 
    m_hub(new OpticksHub(m_ok, true)),    // true: configure and loadGeometry immediately, otherwise too late for CG4
    m_idx(new OpticksIdx(m_hub)),
    m_g4(new CG4(m_hub)),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx)),
#ifdef WITH_OPTIX
    m_ope(new OpEngine(m_hub)),
    m_opv(m_viz ? new OpViz(m_ope,m_viz) : NULL),
#endif
    m_placeholder(0)
{
    init();
    LOG(fatal) << "OKG4Mgr::OKG4Mgr DONE" ;  
}

void OKG4Mgr::init()
{
    LOG(fatal) << "OKG4Mgr::init" ;  

    if(m_viz) m_hub->configureState(m_viz->getSceneConfigurable()) ;    // loads/creates Bookmarks

    m_g4->configure();                     // currently does nothing 

    m_g4->initialize();                   // runManager initialize


    m_hub->overrideMaterialMapA( m_g4->getMaterialMap(), "OKG4Mgr::init/g4mm") ;  // for translation of material indices into GPU texture lines
 
 
    if(m_viz) m_viz->prepareScene();      // setup OpenGL shaders and creates OpenGL context (the window)
 
    // m_hub->loadGeometry();                // creates GGeo instance, loads, potentially modifies for (--test) and registers geometry

    if(m_viz) m_viz->uploadGeometry();    // Scene::uploadGeometry, hands geometry to the Renderer instances for upload

#ifdef WITH_OPTIX
    m_ope->prepareOptiX();                // creates OptiX context and populates with geometry by OGeo, OScintillatorLib, ... convert methods 

    if(m_opv) m_opv->prepareTracer();     // creates ORenderer, OTracer
#endif

    LOG(fatal) << "OKG4Mgr::initGeometry DONE" ;
}


void OKG4Mgr::propagate()
{
    LOG(fatal) << "OKG4Mgr::propagate" ;

    m_g4->propagate();

    NPY<float>* gsgen = m_g4->getGenstepsGenerated();
    NPY<float>* gsrec = m_g4->getGenstepsRecorded();

    int n_gsgen = gsgen ? gsgen->getNumItems() : -1 ; 
    int n_gsrec = gsrec ? gsrec->getNumItems() : -1 ;

    LOG(info) << "OKG4Mgr::propagate"
               << " n_gsgen " <<  n_gsgen  
               << " n_gsrec " <<  n_gsrec 
               ;

    NPY<float>* gs = NULL ; 
    if( n_gsrec == 0 && n_gsgen > 0)
    {
        LOG(fatal) << "no recorded gensteps from G4 but there are generated gs (probably torch running) " ; 
        gs = gsgen ;
    }
    else if( n_gsrec > 0 && n_gsgen == -1)
    {
        LOG(fatal) << "recorded gensteps from G4 and no generated gs (probably g4gun running) " ; 
        gs = gsrec ; 
    } 
    else
    {
        LOG(fatal) << "OKG4Mgr::propagate"
                     << " SKIPPING as no collected optical gensteps (ie Cerenkov or scintillation gensteps) "
                     << " or generated torch gensteps  "
                     ;
        return ;  
    }

    
    // m_hub->translateGensteps(gs);     
    // relabel and apply lookup,  is this needed for both gs flavors 
    // can it move inside m_hub ?


    m_hub->initOKEvent(gs);           // make a new evt 

    if(m_viz)
    { 
        // handling target option inside Scene is inconvenient  TODO: centralize
        int target = m_viz->getTarget();  // hmm target could be non oglrap- specific 
        if(target == 0) 
            m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

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
        m_ope->downloadEvt();
        m_idx->indexEvtOld();

        m_hub->save();
    }
#endif
    LOG(fatal) << "OKG4Mgr::propagate DONE" ;
}




void OKG4Mgr::indexPropagation()
{
    OpticksEvent* evt = m_hub->getOKEvent();
    assert(evt->isOK()); // is this always so ?
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

