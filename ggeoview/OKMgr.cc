#include "OKMgr.hh"

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


bool OKMgr::hasOpt(const char* name){ return m_ok->hasOpt(name); }


OKMgr::OKMgr(int argc, char** argv) 
    :
    m_ok(new Opticks(argc, argv, false)),   // false: NOT integrated running
    m_hub(new OpticksHub(m_ok, true)),      // true: immediate configure and loadGeometry 
    m_idx(new OpticksIdx(m_hub)),
    m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
#ifdef WITH_OPTIX
    m_ope(new OpEngine(m_hub, true)),
    m_opv(m_viz ? new OpViz(m_ope,m_viz, true) : NULL),
#endif
    m_placeholder(0)
{
    init();
    LOG(fatal) << "OKMgr::OKMgr DONE" ;
}

void OKMgr::init()
{
    bool g4gun = m_ok->getSourceCode() == G4GUN ;
    if(g4gun)
         LOG(fatal) << "OKMgr doesnt support G4GUN, other that via loading (TO BE IMPLEMENTED) " ;
    assert(!g4gun);


}


void OKMgr::action()
{
    LOG(fatal) << "OKMgr::action" ; 
    if(hasOpt("load"))
    {
        loadPropagation();
    }
    else if(hasOpt("nopropagate"))
    {
        LOG(info) << "--nopropagate/-P" ;
    }
    else
    { 
        NPY<float>* gs = m_hub->getGensteps();  
        propagate(gs);
    }
}



void OKMgr::propagate(NPY<float>* genstep)
{
    m_hub->initOKEvent(genstep);

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

        m_hub->save();
    }
#endif
}

void OKMgr::indexPropagation()
{
    OpticksEvent* evt = m_hub->getOKEvent();
    if(!evt->isIndexed())
    {
#ifdef WITH_OPTIX 
        m_ope->indexSequence();
#endif
        m_idx->indexBoundariesHost();
    }
    if(m_viz) m_viz->indexPresentationPrep();
}

void OKMgr::loadPropagation()
{
    LOG(fatal) << "OKMgr::loadPropagation" ; 
    m_hub->loadPersistedEvent(); 

    indexPropagation();

    if(!m_viz) return  ;

    int target = m_viz->getTarget();  // hmm target could be non oglrap- specific 
    if(target == 0) 
         m_hub->target();           // if not Scene targetted, point Camera at gensteps of last created evt

    m_viz->uploadEvent();  

    LOG(fatal) << "OKMgr::loadPropagation DONE" ; 
}

void OKMgr::visualize()
{
    if(!m_viz) return ; 
    m_viz->prepareGUI();
    m_viz->renderLoop();    
}

void OKMgr::cleanup()
{
#ifdef WITH_OPTIX
    if(m_ope) m_ope->cleanup();
#endif
    m_hub->cleanup();
    if(m_viz) m_viz->cleanup();
    m_ok->cleanup(); 
}

void OKMgr::dbgSeed()
{
#ifdef WITH_OPTIX
    if(!m_ope) return ; 
    OpticksEvent* evt = m_hub->getOKEvent();    
    NPY<float>* ox = evt->getPhotonData();
    assert(ox);

    // this split between interop and compute mode
    // is annoying, maybe having a shared base protocol
    // for OpEngine and OpticksViz
    // could avoid all the branching 

    if(m_viz) 
    { 
        LOG(info) << "OKMgr::debugSeed (interop) download photon seeds " ;
        m_viz->downloadData(ox) ; 
        ox->save("$TMP/dbgseed_interop.npy");
    }
    else
    {
        LOG(info) << "OKMgr::debugSeed (compute) download photon seeds " ;
        m_ope->downloadPhotonData();  
        ox->save("$TMP/dbgseed_compute.npy");
    }  
#endif
}



